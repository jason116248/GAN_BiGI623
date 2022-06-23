import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNN import GNN
from model.GNN2 import GNN2
from model.AttDGI import AttDGI
from model.pretrained_ml100k import pretrained, genreEncoder, genreDecoder
from model.myDGI import myDGI

class EmbeddingLayer(nn.Module):
    def __init__(self, opt):
        super(EmbeddingLayer, self).__init__()
        self.user_embedding = nn.Embedding(opt["number_user"], opt["feature_dim"])
        self.item_embedding = nn.Embedding(opt["number_item"], opt["feature_dim"])
        self.user_embed = nn.Linear(opt["feature_dim"], opt["hidden_dim"])
        self.item_embed = nn.Linear(opt["feature_dim"], opt["hidden_dim"])
        self.item_index = torch.arange(0, opt["number_item"], 1)
        self.user_index = torch.arange(0, opt["number_user"], 1)

        # pretrained_filter = pretrained(genreEncoder(), genreDecoder())
        # pretrained_filter.load_state_dict(torch.load('./load_pt/pretrained_filter.pt'))
        # with open('./load_pt/user_feature_init', 'rb') as fp:
        #     init_user_feature = pickle.load(fp) # type: list
        # user_feature = []
        # for i in range(len(init_user_feature)):
        #     output = pretrained_filter(torch.tensor(init_user_feature[i], dtype=torch.float).unsqueeze(0))
        #     user_feature.append(output) # list of tensors
        # user_feature = torch.stack(user_feature) 
        # user_feature = torch.squeeze(user_feature, 1) # (943, 18) 
        # user_embedding_parameter = nn.Parameter(user_feature)
        # self.user_embedding.weight = user_embedding_parameter

        if opt["cuda"]:
            self.item_index = self.item_index.cuda()
            self.user_index = self.user_index.cuda()  
        self.GNN = GNN(opt)
    
    def forward(self, ufea, vfea, UV_adj, VU_adj, adj):
        # ufea = self.user_embed(ufea)
        # vfea = self.item_embed(vfea)
        learn_user,learn_item = self.GNN(ufea,vfea,UV_adj,VU_adj,adj)
        return learn_user,learn_item    
        
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt=opt
        self.embedding = EmbeddingLayer(opt)
        self.GNN = GNN(opt) # fast mode(GNN), slow mode(GNN2)
        self.extractor = myDGI(opt)
        self.dropout = opt["dropout"]

    def score_predict(self, fea):
        out = self.embedding.GNN.score_function1(fea) #[128, 1650, 32]
        out = F.relu(out)
        out = self.embedding.GNN.score_function2(out)
        out = F.relu(out)
        out = self.embedding.GNN.score_function3(out)
        out = torch.sigmoid(out)
        return out.view(out.size()[0], -1)

    def score(self, fea):
        out = self.embedding.GNN.score_function1(fea)           
        out = F.relu(out)
        out = self.embedding.GNN.score_function2(out)          
        out = F.relu(out)     
        out = self.embedding.GNN.score_function3(out)
        out = torch.sigmoid(out)
        return out.view(-1)

    def getGlobalRepr(self):
        return self.extractor.global_repr

    def forward(self, user_hidden_out, item_hidden_out, fake_user_hidden_out, fake_item_hidden_out,
                UV, VU, CUV, CVU, user_One, item_One, UV_rated, VU_rated, relation_UV_adj, relation_VU_adj):
        
        return self.extractor(user_hidden_out, item_hidden_out, fake_user_hidden_out, fake_item_hidden_out,
                            UV, VU, CUV, CVU, user_One, item_One,
                            UV_rated, VU_rated, relation_UV_adj, relation_VU_adj)
                

class Discriminator(nn.Module):
    def __init__(self, d_model) -> None:
        super(Discriminator, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1)
        self.Encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.lin = nn.Linear(d_model, 1)
        self.sigm = nn.Sigmoid()
    def forward(self, vector): #vector: [128, 64+64]
        vector = torch.unsqueeze(vector, 0)
        output = self.Encoder(vector)
        output = torch.squeeze(output, 0)
        output = self.lin(output)
        score = self.sigm(output)
        return score        