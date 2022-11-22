import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt          
        self.multi = multi_layer            #5
        self.weight = nn.Parameter(
            torch.Tensor(3 + 1, 1, self.time_step * self.multi,
                         self.multi * self.time_step))  # [K+1, 1, in_c, out_c]     torch.Size([1, 4, 1, 60, 60])
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)          #60, 12
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi            #4 * 5 = 20
        for i in range(3):
            print(f"self.GLUs i : {i}")
            if i == 0:
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))     # 12 * 4 = 48, 12 * 20 = 240
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
            else:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))

    def spe_seq_cell(self, input):
        batch_size, k, input_channel, node_cnt, time_step = input.size()        # torch.Size([32, 4, 1, 140, 12])
        input = input.view(batch_size, -1, node_cnt, time_step)                 # torch.Size([32, 4, 140, 12])
        # print("input shape : ",input.shape)
        # ffted = torch.rfft(input, 1, onesided=False)
        ffted = torch.fft.fft(input, None, -1) #, onesided=False)          # torch.Size([32, 4, 140, 12])
        # print(torch.view_as_real(ffted).shape)
        ffted = torch.view_as_real(ffted)       #version difference         torch.Size([32, 4, 140, 12, 2])
        # print("ffted shape : ",ffted.shape)
        # print(list(ffted[:,:,:,0][0].shape))
        # print(list(ffted[...,0][0].shape))
        # print(ffted[..., 0].shape)
        # print(ffted[..., 1].shape)
        temp = ffted[..., 0]
        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)     #실수부 real view_as_real 처음꺼
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)      #허수부 img
        
        # print(real.shape)
        
        for i in range(3):
            real = self.GLUs[i * 2](real)       #real : torch.Size([32, 140, 48]) -> torch.Size([32, 140, 240])
            img = self.GLUs[2 * i + 1](img)
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()       #torch.Size([32, 140, 4, 60]) -> torch.Size([32, 4, 140, 60])
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()         #torch.Size([32, 4, 140, 60])
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)         #torch.Size([32, 4, 140, 60, 2])
        iffted = torch.fft.irfft(time_step_as_inner, 1) #, onesided=False)                      #torch.Size([32, 4, 140, 60, 1])
        return iffted.squeeze(-1)

    def forward(self, x, mul_L):        #mul_L : [4, 140, 140], x : [32, 1, 140, 12]
        mul_L = mul_L.unsqueeze(1)      #mul_L : [4, 1, 140, 140]
        x = x.unsqueeze(1)              #x : [32, 1, 1, 140, 12]        1 -> 4로 증가
        gfted = torch.matmul(mul_L, x)  #gfted : [32, 4, 1, 140, 12]
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)         #gconv_input : torch.Size([32, 4, 1, 140, 60, 1])
        # print(f"gconv size : gconv_input")
        igfted = torch.matmul(gconv_input, self.weight)           #self.weight : torch.Size([1, 4, 1, 60, 60])
        # igfted = torch.matmul(self.weight, gconv_input)             #self.weight : torch.Size([1, 4, 1, 60, 60])
        # igfted = torch.sum(igfted, dim=1)                           #torch.Size([32, 4, 140, 60, 1])        32, 1 이여야하는데
        igfted = torch.mean(igfted, dim=1)                           #torch.Size([32, 4, 140, 60, 1])        32, 1 이여야하는데
        igfted = torch.squeeze(igfted, dim=-1)                           #* torch.Size([32, 4, 140, 60]) 추가
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)            #torch.Size([32, 4, 140, 12])
        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)  #time_step -> time_step  #torch.Size([32, 1, 140, 12])
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)     
        else:
            backcast_source = None
        return forecast, backcast_source        #torch.Size([32, 4, 140, 12]), torch.Size([32, 4, 140, 12])


class Model(nn.Module):
    def __init__(self, units, stack_cnt, time_step, multi_layer, horizon=1, dropout_rate=0.5, leaky_rate=0.2,
                 device='cpu'):
        super(Model, self).__init__()
        self.unit = units
        self.stack_cnt = stack_cnt          #stack_cnt = 2
        self.unit = units
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.time_step, self.unit)
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i) for i in range(self.stack_cnt)])
        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x):
        mul_L, attention = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        result = []
        for stack_i in range(self.stack_cnt):
            # print(stack_i)
            forecast, X = self.stock_block[stack_i](X, mul_L)       #torch.Size([32, 4, 140, 12]), torch.Size([32, 4, 140, 12])
            result.append(forecast)
        forecast = result[0] + result[1]
        forecast = self.fc(forecast)
        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1), attention
        else:
            return forecast.permute(0, 2, 1).contiguous(), attention
