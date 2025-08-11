import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

LABELS = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]


class Model3DResnet(nn.Module):
    def __init__(self, in_channels=1,num_labels=7, num_classes=3, pretrained=True, freeze_backbone=True, dropout_p=0.3,freeze_n=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.freeze_n = freeze_n
        # 🔍 Backbone preentrenado (3D ResNet-18)
        self.backbone = r3d_18(pretrained=pretrained)
        self.freeze_resnet3d_stages(freeze_n=freeze_n)
        # 🧱 Adaptador para imágenes monocanal (MRI/CT)
        if in_channels != 3:
            #self.adapter = nn.Conv3d(in_channels, 3, kernel_size=1)
            """
            self.adapter = nn.Sequential(
                nn.Conv3d(in_channels, 3, kernel_size=1),
                #nn.Dropout3d(p=0.2)
            )
            """
            self.adapter = nn.Sequential(
                nn.Conv3d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(16, 3, kernel_size=1)
            )

        else:
            self.adapter = nn.Identity()

        # 🧠 Congelar el backbone si se desea
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 🎯 Head personalizada con Dropout
        in_features = self.backbone.fc.in_features
        """
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_labels * num_classes)
        )
        """
        in_features = self.backbone.fc.in_features  # original
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features + 3, in_features),  # +3 por el one-hot
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_labels * num_classes)
        )


    #def forward(self, x):
    #    x = self.adapter(x)         # (B, 1, D, H, W) → (B, 3, D, H, W)
        # En lugar de usar Conv3d, simplemente repite el canal:
        #x = x.repeat(1, 3, 1, 1, 1)  # (B, 1, D, H, W) → (B, 3, D, H, W)
    #    return self.backbone(x).view(-1, self.num_labels, self.num_classes)

    def forward(self, x, view_onehot):
        x = self.adapter(x)          # (B, 1, D, H, W) → (B, 3, D, H, W)
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.flatten(1)             # (B, F)

        # 👇 Concatena view embedding
        x = torch.cat([x, view_onehot], dim=1)  # (B, F + 3)

        # 🎯 Pasa por la head
        x = self.backbone.fc(x)     # fc ahora espera (F + 3)
        return x.view(-1, self.num_labels, self.num_classes)


    def freeze_resnet3d_stages(self, freeze_n=2):
        """
        Congela los primeros N bloques de un modelo basado en r3d_18.

        Args:
            model: instancia de Videor3d18Classifier
            freeze_n: cuántos bloques congelar (0 = nada, 5 = todo)
        """
        assert hasattr(self.backbone, 'stem'), "El modelo no parece ser un ResNet3D"
        assert 0 <= freeze_n <= 5, "freeze_n debe estar entre 0 y 5"

        # Lista ordenada de bloques
        blocks = [
            self.backbone.stem,   # 0
            self.backbone.layer1, # 1
            self.backbone.layer2, # 2
            self.backbone.layer3, # 3
            self.backbone.layer4  # 4
        ]

        # Congelar los primeros `freeze_n` bloques
        for i in range(freeze_n):
            for param in blocks[i].parameters():
                param.requires_grad = False
            print(f"❄️  Bloque {i} ({blocks[i]._get_name()}) congelado")

        # Descongelar los siguientes (por si fue llamado luego de un freeze total)
        for i in range(freeze_n, len(blocks)):
            for param in blocks[i].parameters():
                param.requires_grad = True
            print(f"🔓 Bloque {i} ({blocks[i]._get_name()}) entrenable")

        # Asegura que la head (fc) siempre esté entrenable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

