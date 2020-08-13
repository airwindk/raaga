import torch.nn as nn
import torch
from collections import OrderedDict
import config

class RaagaModel(nn.Module):
    def __init__(self, dropout=0.15, hidden_size=256):
        super(RaagaModel, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(1)),

            ('conv1', nn.Conv2d(1, 64, 3, padding=1)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.LeakyReLU()),
            ('pool1', nn.MaxPool2d([1, 2])),
            ('drop1', nn.Dropout(p=dropout)),

            ('conv2', nn.Conv2d(64, 128, 3, padding=1)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.LeakyReLU()),
            ('pool2', nn.MaxPool2d([1, 3])),
            ('drop2', nn.Dropout(p=dropout)),

            ('conv3', nn.Conv2d(128, 150, 3, padding=1)),
            ('norm3', nn.BatchNorm2d(150)),
            ('relu3', nn.LeakyReLU()),
            ('pool3', nn.MaxPool2d([4, 2])),
            ('drop3', nn.Dropout(p=dropout)),

            ('conv4', nn.Conv2d(150, 200, 3, padding=1)),
            ('norm4', nn.BatchNorm2d(200)),
            ('gba', nn.AvgPool2d([3, len(config.labels_map['raaga'])*2])),
            ('drop4', nn.Dropout(p=dropout))
        ]))

        self.fc1 = nn.Linear(200, len(config.labels_map['raaga']))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.encoder(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        return x

    def summary(self, input_size):
       	def register_hook(module):
            def hook(module, input, output):
                if module._modules:  # only want base layers
                    return
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)
                m_key = '%s-%i' % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = None
                if output.__class__.__name__ == 'tuple':
                    summary[m_key]['output_shape'] = list(output[0].size())
                else:
                    summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = None

                params = 0
                # iterate through parameters and count num params
                for name, p in module._parameters.items():
                    params += torch.numel(p.data)
                    summary[m_key]['trainable'] = p.requires_grad

                summary[m_key]['nb_params'] = params

            if not isinstance(module, torch.nn.Sequential) and \
               not isinstance(module, torch.nn.ModuleList) and \
               not (module == self):
                hooks.append(module.register_forward_hook(hook))

        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [(torch.rand(1, *in_size)) for in_size in input_size]
        else:
            x = (torch.randn(1, *input_size))

        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        self.apply(register_hook)
        # make a forward pass
        self(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        # print out neatly
        def get_names(module, name, acc):
            if not module._modules:
                acc.append(name)
            else:
                for key in module._modules.keys():
                    p_name = key if name == "" else name + "." + key
                    get_names(module._modules[key], p_name, acc)
        names = []
        get_names(self, "", names)

        col_width = 25  # should be >= 12
        summary_width = 61

        def crop(s):
            return s[:col_width] if len(s) > col_width else s

        print('_' * summary_width)
        print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
            'Layer (type)', 'Output Shape', 'Param #', col_width))
        print('=' * summary_width)
        total_params = 0
        trainable_params = 0
        for (i, l_type), l_name in zip(enumerate(summary), names):
            d = summary[l_type]
            total_params += d['nb_params']
            if 'trainable' in d and d['trainable']:
                trainable_params += d['nb_params']
            print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
                crop(l_name + ' (' + l_type[:-2] + ')'), crop(str(d['output_shape'])),
                crop(str(d['nb_params'])), col_width))
            if i < len(summary) - 1:
                print('_' * summary_width)
        print('=' * summary_width)
        print('Total params: ' + str(total_params))
        print('Trainable params: ' + str(trainable_params))
        print('Non-trainable params: ' + str((total_params - trainable_params)))
        print('_' * summary_width)

# import torch
# torch.manual_seed(123)
# import torch.nn as nn
# import config

# class RaagaModel(nn.Module):
#     def __init__(self):
#         super(RaagaModel, self).__init__()

#         self._extractor = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),

#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),

#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=4),

#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=4)
#         )

#         self._classifier = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
#                                          nn.ReLU(),
#                                          nn.Dropout(),
#                                          nn.Linear(in_features=1024, out_features=256),
#                                          nn.ReLU(),
#                                          nn.Dropout(),
#                                          nn.Linear(in_features=256, out_features=len(config.raaga_map)))
#         self.apply(self._init_weights)

#     def forward(self, x):
#         x = torch.unsqueeze(x,1)
        
#         print(x.shape)
        
#         x = self._extractor(x)
#         x = x.view(x.size(0), -1)
#         score = self._classifier(x)
#         return score

#     def _init_weights(self, layer) -> None:
#         if isinstance(layer, nn.Conv1d):
#             nn.init.kaiming_uniform_(layer.weight)
#         elif isinstance(layer, nn.Linear):
#             nn.init.xavier_uniform_(layer.weight)