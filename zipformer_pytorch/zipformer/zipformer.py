import torch
from torch import nn, einsum
import torch.nn.functional as F

try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
except ImportError:
    def rearrange(tensor, pattern, **sizes):
        if pattern == 'b c t m -> b t (c m)':
            b, c, t, m = tensor.shape
            return tensor.permute(0, 2, 1, 3).reshape(b, t, c * m)
        if pattern == 'b n (h d) -> b h n d':
            h = sizes["h"]
            b, n, hd = tensor.shape
            return tensor.reshape(b, n, h, hd // h).permute(0, 2, 1, 3)
        if pattern == 'b h n d -> b n (h d)':
            b, h, n, d = tensor.shape
            return tensor.permute(0, 2, 1, 3).reshape(b, n, h * d)
        if pattern == 'i -> i ()':
            return tensor.unsqueeze(1)
        if pattern == 'j -> () j':
            return tensor.unsqueeze(0)
        if pattern == 'b i -> b () i ()':
            return tensor.unsqueeze(1).unsqueeze(-1)
        if pattern == 'b j -> b () () j':
            return tensor.unsqueeze(1).unsqueeze(1)
        raise NotImplementedError(f"Unsupported rearrange pattern without einops: {pattern}")


    class Rearrange(nn.Module):
        def __init__(self, pattern):
            super().__init__()
            self.pattern = pattern

        def forward(self, x):
            if self.pattern == 'b n c -> b c n':
                return x.permute(0, 2, 1)
            if self.pattern == 'b c n -> b n c':
                return x.permute(0, 2, 1)
            raise NotImplementedError(
                f"Unsupported Rearrange pattern without einops: {self.pattern}"
            )

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

class SwooshR(nn.Module):
    def forward(self,x):
        return torch.log(1+torch.exp(x-1)) - 0.08*x - 0.313261687

class SwooshL(nn.Module):
    def forward(self,x):
        return torch.log(1+torch.exp(x-4)) - 0.08*x - 0.035
    
class BiasNorm(nn.Module):
    def __init__(self, num_features):
        super(BiasNorm, self).__init__()
        self.num_features = num_features
        self.bias = nn.Parameter(torch.ones(1, 1,num_features))  # Adjusting the shape of the bias tensor
        self.weight = nn.Parameter(torch.zeros(1, 1,num_features))
        

    def forward(self, x, mask = None):
        biased_x = x - self.bias
        squared = biased_x.pow(2)
        if exists(mask):
            mask = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            squared = squared * mask
            denom = (mask.sum(dim = -2, keepdim = True) * x.size(-1)).clamp_min(1.0)
            mean_squared = squared.sum(dim = -1, keepdim = True).sum(dim = -2, keepdim = True) / denom
        else:
            mean_squared = squared.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        rms = torch.sqrt(mean_squared)
        return (x / rms) * torch.exp(self.weight)
    

class Downsample(nn.Module):
    def __init__(self, num_channels):
        super(Downsample, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_channels, 2))

    def forward(self, x):
        # weighted average of frames
        weights = F.softmax(self.weights, dim=1)
        x = x.unbind(dim=1)  
        x = [(x[i]*weights[i%self.weights.shape[0], 0] + x[i+1]*weights[i%self.weights.shape[0], 1])/2 for i in range(0, len(x)-1, 2)]
        x = torch.stack(x, dim=1)
        return x

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x):
        # Repeat each frame twice along the time dimension
        return x.repeat(1, 1, 2).view(x.size(0), -1, x.size(2))


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

class ByPass(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.num_features = num_features
        self.c = nn.Parameter(torch.ones(1,1,self.num_features))

    def forward(self,x,y):
        return (1.0 - self.c) * x + self.c * y

class ConvNextLayer(nn.Module):
    def __init__(self,dim):
        super(ConvNextLayer,self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs
        self.act = SwooshL()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self,x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x # residual connection
        return x


class ConvEmbed(nn.Module):
    def __init__(self, in_channels, out_channels=[8, 32, 128], kernel_sizes=[(3, 3), (3, 3), (3, 3)], strides=[(1, 2), (2, 2), (1, 2)]):
        super(ConvEmbed, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1,out_channels[0],kernel_sizes[0], stride=strides[0], padding=(1, 1)),
            nn.Conv2d(out_channels[0],out_channels[1],kernel_sizes[1], stride=strides[1], padding=(1, 1)),
            nn.Conv2d(out_channels[1],out_channels[2],kernel_sizes[2], stride=strides[2], padding=(1, 1))
        )
        self.convnext_layer = ConvNextLayer(in_channels)
        self.linear = nn.Linear(in_features=in_channels,out_features=in_channels)
        self.norm = BiasNorm(in_channels)
        self.act = SwooshR()
        self.adjust_shape = nn.ModuleDict()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.convnext_layer(x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        batch_size, num_channels, num_time_steps, mel_features = x.shape
        x = rearrange(x, 'b c t m -> b t (c m)')
        
        key = f"{num_channels}_{mel_features}"
        # This linear layer is not properly understood,its description from Nextformer paper is - 
        # linear layer to convert channel-timefrequency output to time-channel output for subsequent Conformers.
        if key not in self.adjust_shape:
            self.adjust_shape[key] = nn.Linear(num_channels * mel_features, num_channels)

        x = self.adjust_shape[key](x)
        x = self.norm(x)
        return x
    

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.linear_1 = nn.Linear(dim,dim*mult)
        self.act_1 = SwooshL()
        self.drp_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim * mult, dim)
        self.act_2 = SwooshL()
        self.drp_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act_1(x)
        x = self.drp_1(x)
        x = self.linear_2(x)
        x = self.act_2(x)
        x = self.drp_2(x)
        return x

class ZipformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.norm = BiasNorm(dim)
        self.to_channels = Rearrange('b n c -> b c n')
        self.pointwise_in = nn.Conv1d(dim, inner_dim * 2, 1)
        self.act_in = SwooshR()
        self.depthwise = DepthWiseConv1d(inner_dim*2, inner_dim*2, kernel_size = kernel_size, padding = padding)
        self.act_depthwise = SwooshR()
        self.pointwise_out = nn.Conv1d(inner_dim*2, dim, 1)
        self.to_sequence = Rearrange('b c n -> b n c')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        x = self.norm(x, mask = mask)
        x = self.to_channels(x)
        x = self.pointwise_in(x)
        x = self.act_in(x)
        x = self.depthwise(x)
        x = self.act_depthwise(x)
        x = self.pointwise_out(x)
        x = self.to_sequence(x)
        x = self.dropout(x)
        return x



class MultiHeadAttentionWeight(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim , bias = False)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None
    ):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, k  = self.to_q(x), self.to_k(context)
        q, k  = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding

        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        return attn

class SelfAttention(nn.Module):
    def __init__(self,dim,heads = 8,dim_head = 64,dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,attn_wts):
        v = self.to_v(x)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        out = einsum('b h i j, b h j d -> b h i d', attn_wts, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class NonLinearAttention(nn.Module):
    def __init__(self, hidden_dim,heads=8):
        super(NonLinearAttention, self).__init__()
        self.A = nn.Linear(hidden_dim, hidden_dim * 3 // 4)
        self.B = nn.Linear(hidden_dim, hidden_dim * 3 // 4)
        self.C = nn.Linear(hidden_dim, hidden_dim * 3 // 4)
        self.linear = nn.Linear(hidden_dim * 3 // 4, hidden_dim)
        self.heads = heads

    def forward(self, x, attention_weights):
        A = self.A(x)
        B = self.B(x)
        C = self.C(x)
        A = rearrange(A, 'b n (h d) -> b h n d', h = self.heads)
        B = rearrange(B, 'b n (h d) -> b h n d', h = self.heads)
        C = rearrange(C, 'b n (h d) -> b h n d', h = self.heads)
    
        B = torch.tanh(B)
        
        B = einsum('b h i j, b h j d -> b h i d', attention_weights, B)
        
        output = C * B
        
        output = output * A
        output = rearrange(output, 'b h n d -> b n (h d)')

        output = self.linear(output)
        
        return output

class ZipformerBlock(nn.Module):
    def __init__(self,dim,heads = 8,mult = 4):
        """
        dim : embedding dim for Zipformer Block
        heads : num of heads for multihead attention
        mult : multiplying factor for the hidden dimension for Feed Forward Block
        """
        super(ZipformerBlock,self).__init__()
        self.ff1 = FeedForward(dim,mult)
        self.ff2 = FeedForward(dim,mult)
        self.ff3 = FeedForward(dim,mult)
        self.mhaw = MultiHeadAttentionWeight(dim,heads)
        self.nla = NonLinearAttention(dim,heads)
        self.sa1 = SelfAttention(dim,heads)
        self.sa2 = SelfAttention(dim,heads)
        self.conv1 = ZipformerConvModule(dim)
        self.conv2 = ZipformerConvModule(dim)
        self.byp1 = ByPass(dim)
        self.byp2 = ByPass(dim)
        self.norm = BiasNorm(dim)

    def forward(self, x, mask = None):
        if exists(mask):
            mask_expanded = mask.unsqueeze(-1)
            x = x.masked_fill(~mask_expanded, 0.0)
        else:
            mask_expanded = None
        inp = x
        x = x + self.ff1(x)
        if exists(mask_expanded):
            x = x.masked_fill(~mask_expanded, 0.0)
        attn_wts = self.mhaw(x, mask = mask)
        x = x + self.nla(x,attn_wts)
        if exists(mask_expanded):
            x = x.masked_fill(~mask_expanded, 0.0)
        x = x + self.sa1(x,attn_wts)
        if exists(mask_expanded):
            x = x.masked_fill(~mask_expanded, 0.0)
        x = x + self.conv1(x, mask = mask)
        if exists(mask_expanded):
            x = x.masked_fill(~mask_expanded, 0.0)
        x = x + self.ff2(x)
        if exists(mask_expanded):
            x = x.masked_fill(~mask_expanded, 0.0)
        x = self.byp1(inp,x)
        if exists(mask_expanded):
            x = x.masked_fill(~mask_expanded, 0.0)
        x = x + self.sa2(x,attn_wts)
        if exists(mask_expanded):
            x = x.masked_fill(~mask_expanded, 0.0)
        x = x + self.conv2(x, mask = mask)
        if exists(mask_expanded):
            x = x.masked_fill(~mask_expanded, 0.0)
        x = x + self.ff3(x)
        if exists(mask_expanded):
            x = x.masked_fill(~mask_expanded, 0.0)
        x = self.norm(x, mask = mask)
        x = self.byp2(inp,x)
        if exists(mask_expanded):
            x = x.masked_fill(~mask_expanded, 0.0)
        return x

class Zipformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convembed = ConvEmbed(128)
        self.initial_zipformer_blocks = nn.Sequential(
            ZipformerBlock(128,2),
            ZipformerBlock(128,2)
        )
        self.linear_1 = nn.Linear(128,192) # to adjust the embedding dims for Zipformer
        self.first_block = nn.Sequential(
            Downsample(128),
            ZipformerBlock(192,2),
            ZipformerBlock(192,2),
            Upsample()
        )
        self.byp1 = ByPass(192)
        self.linear2 = nn.Linear(192,256) # we can add more number of linear layers to adjust our hidden and embedding dims
        self.second_block = nn.Sequential(
            Downsample(256),
            ZipformerBlock(256,2),
            ZipformerBlock(256,2),
            Upsample()
        )
        self.byp2 = ByPass(256)
        self.third_block = nn.Sequential(
            Downsample(256),
            ZipformerBlock(256,2),
            ZipformerBlock(256,2),
            Upsample()
        )
        self.byp3 = ByPass(256)
        self.fourth_block = nn.Sequential(
            Downsample(256),
            ZipformerBlock(256,2),
            ZipformerBlock(256,2),
            Upsample()
        )
        self.byp4 = ByPass(256)
        self.fifth_block = nn.Sequential(
            Downsample(256),
            ZipformerBlock(256,2),
            ZipformerBlock(256,2),
            Upsample()
        )
        self.bpyp5 = ByPass(256)
        self.final_downsample = Downsample(256)

    def forward(self,x):
        x = self.convembed(x)
        x = self.initial_zipformer_blocks(x)
        x = self.linear_1(x)
        x1 = x
        for _ in range(2):
            x = self.first_block(x)
        x = self.byp1(x1,x)
        x = self.linear2(x)
        x2 = x
        for _ in range(3):
            x = self.second_block(x)
        x = self.byp2(x2,x)
        x3 = x
        for _ in range(4):
            x = self.third_block(x)
        x = self.byp3(x3,x)
        x4 = x
        for _ in range(3):
            x = self.fourth_block(x)
        x = self.byp4(x4,x)
        x5 = x
        for _ in range(2):
            x = self.fifth_block(x)
        x = self.bpyp5(x5,x)
        x = self.final_downsample(x)
        return x
