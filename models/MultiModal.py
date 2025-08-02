import torch
from torch import nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from .Attention import SelfAttention, CrossAttention
from plink.genome_local_net import DeepcvGRS            
 
 
def copy_weights(source_model, target_model):
    with torch.no_grad():
        for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
            target_param.copy_(source_param)

            
class BaseMultiModalNet(nn.Module):
    def __init__(self, snp_number: int, image_features_num: int = 768, gene_features_num: int = 64):
        super(BaseMultiModalNet, self).__init__()
        gene_features = DeepcvGRS(snp_number)
        self.SNP = DeepcvGRS(snp_number)
        #copy_weights(gene_features,  self.SNP)
        del gene_features.net[-1]
        self.out_feature =gene_features.out_feature
        self.gene_features =nn.Sequential(
             gene_features,# nn.BatchNorm1d(self.out_feature),
             nn.ReLU(),
        )
        image_model = convnext_tiny(num_classes=1)#num_classes=1 0.9295  #
        self.IMAGE = convnext_tiny(num_classes=1)
        copy_weights(image_model, self.IMAGE)   
        del image_model.classifier[-1]    
        self.IMAGE.classifier[-1] = nn.Linear(768, 1)
        self.image_features = nn.Sequential(
            image_model,
            nn.BatchNorm1d(image_features_num),
            nn.ReLU(),
        )
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ConcatMultiModalNet(BaseMultiModalNet):
    def __init__(self, snp_number: int, image_features_num: int = 768, gene_features_num: int = 64,
                 fusion_features_num: int = 32):
        super(ConcatMultiModalNet, self).__init__(snp_number, image_features_num, gene_features_num)
        self.concat_fusion = nn.Sequential(
            nn.Linear(image_features_num + self.out_feature, fusion_features_num, bias=True),
            # nn.Linear(image_features_num + gene_features_num, fusion_features_num),
            nn.BatchNorm1d(fusion_features_num),
            nn.ReLU(),
            nn.Linear(fusion_features_num, 1),
        )
        self.apply(self._init_weights)

    # def apply(self, fn):
    #     self.gene_features.apply(fn)
    #     self.concat_fusion.apply(fn)
    def forward(self, snps, image):
        image_feature = self.image_features(image)
        gene_feature = self.gene_features(snps)
        x = torch.cat([image_feature, gene_feature], dim=1)
        y = self.concat_fusion(x)
        return y

class GateKroneckerFusion(BaseMultiModalNet):
    def __init__(self,snp_number, skip=1,gate=1, image_features_num: int = 768, gene_features_num=64,fusion_features_num=32,dim1=32,dim2=32, dropout_rate=0.25,return_only_snp_image=False,drop_snp=False,drop_image=False,calculate_contribution=False):
        super(GateKroneckerFusion, self).__init__(snp_number, image_features_num, gene_features_num)
        self.skip = skip
        self.gate = gate
        self.drop_snp = drop_snp
        self.drop_image = drop_image
        self.calculate_contribution = calculate_contribution
        self.return_only_snp_image=return_only_snp_image
        gene_features_num =self.out_feature
        skip_dim = gene_features_num+image_features_num if skip else 0

        self.linear_snp = nn.Sequential(nn.Linear(gene_features_num, dim1), nn.ReLU())#nn.ReLU
        self.linear_z1 = nn.Sequential(nn.Linear(gene_features_num+image_features_num, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))
               
        self.linear_image = nn.Sequential(nn.Linear(image_features_num, dim2),  nn.ReLU()) 
        self.linear_z2 = nn.Sequential(nn.Linear(gene_features_num+image_features_num, dim1))     
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), fusion_features_num),  nn.BatchNorm1d(fusion_features_num), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Linear(fusion_features_num+skip_dim, 1))

    def forward(self, snps, image):
        snp_output =self.SNP(snps)
        image_output = self.IMAGE(image)
        gene_feature = self.gene_features(snps)
        image_feature = self.image_features(image)
        ### Gated Multimodal Units
        if self.gate:            
            snp = self.linear_snp(gene_feature)
            z1 = self.linear_z1(torch.cat((gene_feature, image_feature), dim=1))
            z1_gate = nn.Sigmoid()(z1)
            self.z1_gate = z1_gate  
            o1 = self.linear_o1(z1_gate*snp)
            
            image = self.linear_image(image_feature)
            z2 = self.linear_z2(torch.cat((gene_feature, image_feature), dim=1))
            z2_gate = nn.Sigmoid()(z2)
            self.z2_gate = z2_gate
            o2 = self.linear_o2(z2_gate*image)
        else:
            snp = self.linear_snp(gene_feature)
            o1 = self.linear_o1(snp)
            image = self.linear_image(image_feature)
            o2 = self.linear_o2(image)

        ### Fusion
        device = o1.device
        o1 = torch.cat((o1, torch.full((o1.shape[0], 1), 1.0,device = device)), dim=1)
        o2 = torch.cat((o2, torch.full((o2.shape[0], 1), 1.0,device = device)), dim=1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.drop_snp:
            drop_out = torch.zeros_like(out)
            drop_gene_feature = torch.zeros_like(gene_feature)
            drop_image_feature = image_feature
            drop_snp_out = torch.cat((drop_out, drop_gene_feature, drop_image_feature), 1) 
            image = self.encoder2(drop_snp_out)
        if self.drop_image:
            drop_out = torch.zeros_like(out)
            drop_gene_feature = gene_feature
            drop_image_feature = torch.zeros_like(image_feature)
            drop_image_out = torch.cat((drop_out, drop_gene_feature, drop_image_feature), 1)
            snp = self.encoder2(drop_image_out)
        if self.skip: out = torch.cat((out, gene_feature, image_feature), 1)
        self.out=out
        snp_image = self.encoder2(out)
                         
        if self.return_only_snp_image:
            return snp_image
        elif self.calculate_contribution:
                return snp_output, image_output, snp_image,snp,image
        else:
            return snp_output, image_output, snp_image
        #return snp_output,image_output,snp_image #snp,image,



class SelfAttentionMultiModalNet(BaseMultiModalNet):
    def __init__(self, snp_number: int, image_features_num: int = 768, gene_features_num: int = 64,
                 fusion_features_num: int = 32):
        super(SelfAttentionMultiModalNet, self).__init__(snp_number, image_features_num, gene_features_num)
        self.image_gene_features=nn.Sequential(
            nn.Linear(image_features_num , gene_features_num, bias=False),
            nn.BatchNorm1d(gene_features_num),
            nn.ReLU(),
        )
        self.self_attention_fusion = nn.Sequential(
            SelfAttention(gene_features_num + gene_features_num),
            nn.Linear(gene_features_num + gene_features_num, fusion_features_num, bias=True),
            nn.BatchNorm1d(fusion_features_num),
            nn.ReLU(),
            nn.Linear(fusion_features_num, 1),
        )
        self.apply(self._init_weights)

    def forward(self, snps, image):
        image_feature = self.image_features(image)
        gene_feature = self.gene_features(snps)
        image_gene_feature=self.image_gene_features(image_feature)
        x = torch.cat([image_gene_feature, gene_feature], dim=1)
        y = self.self_attention_fusion(x)
        return y


class MultiHeadAttentionMultiModalNet(BaseMultiModalNet):
    def __init__(self, snp_number: int, image_features_num: int = 768, gene_features_num: int = 64,
                 fusion_features_num: int = 32):
        super(MultiHeadAttentionMultiModalNet, self).__init__(snp_number, image_features_num, gene_features_num)
        self.image_gene_features = nn.Sequential(
            nn.Linear(image_features_num, gene_features_num, bias=False),
            nn.BatchNorm1d(gene_features_num),
            nn.ReLU(),
        )
        self.multi_head_attention = nn.MultiheadAttention(gene_features_num + gene_features_num, 4)
        self.multi_head_fusion = nn.Sequential(
            nn.Linear(gene_features_num + gene_features_num, fusion_features_num, bias=False),
            nn.BatchNorm1d(fusion_features_num),
            nn.ReLU(),
            nn.Linear(fusion_features_num, 1),
        )
        self.apply(self._init_weights)

    def forward(self, snps, image):
        image_feature = self.image_features(image)
        gene_feature = self.gene_features(snps)
        image_gene_feature = self.image_gene_features(image_feature)
        x = torch.cat([image_gene_feature, gene_feature], dim=1)
        y = self.multi_head_attention(x, x, x)[0]
        y = self.multi_head_fusion(y)
        return y


class TransformerMultiModalNet(BaseMultiModalNet):
    def __init__(self, snp_number: int, image_features_num: int = 768, gene_features_num: int = 64,
                 fusion_features_num: int = 32):
        super(TransformerMultiModalNet, self).__init__(snp_number, image_features_num, gene_features_num)
        self.image_gene_features = nn.Sequential(
            nn.Linear(image_features_num, gene_features_num, bias=False),
            nn.BatchNorm1d(gene_features_num),
            nn.ReLU(),
        )
        self.transformer_fusion = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=gene_features_num + gene_features_num, nhead=4),
                num_layers=2
            ),
            nn.Linear(gene_features_num + gene_features_num, fusion_features_num, bias=False),
            nn.BatchNorm1d(fusion_features_num),
            nn.ReLU(),
            nn.Linear(fusion_features_num, 1),
        )
        self.apply(self._init_weights)

    def forward(self, snps, image):
        image_feature = self.image_features(image)
        gene_feature = self.gene_features(snps)
        image_gene_feature = self.image_gene_features(image_feature)
        x = torch.cat([image_gene_feature, gene_feature], dim=1)
        y = self.transformer_fusion(x)
        return y


class CrossAttentionMultiModalNet(BaseMultiModalNet):
    def __init__(self, snp_number: int, image_features_num: int = 768, gene_features_num: int = 64,
                 fusion_features_num: int = 32):
        super(CrossAttentionMultiModalNet, self).__init__(snp_number, image_features_num, gene_features_num)
        self.image_gene_features = nn.Sequential(
            nn.Linear(image_features_num, gene_features_num, bias=False),
            nn.BatchNorm1d(gene_features_num),
            nn.ReLU(),
        )
        self.cross_attention = CrossAttention(gene_features_num, gene_features_num, gene_features_num)
        self.cross_attention_fusion = nn.Sequential(
            nn.Linear(gene_features_num, fusion_features_num, bias=False),
            nn.BatchNorm1d(fusion_features_num),
            nn.ReLU(),
            nn.Linear(fusion_features_num, 1),
        )
        self.apply(self._init_weights)

    def forward(self, snps, image):
        image_feature = self.image_features(image)
        gene_feature = self.gene_features(snps)
        image_gene_feature = self.image_gene_features(image_feature)
        x = self.cross_attention(gene_feature, image_gene_feature)
        # x=self.cross_attention(gene_feature, image_feature)
        y = self.cross_attention_fusion(x)
        return y
