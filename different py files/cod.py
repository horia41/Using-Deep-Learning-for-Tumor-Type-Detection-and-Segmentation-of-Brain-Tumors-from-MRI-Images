import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision

class FCNResNet50Lightning(pl.LightningModule):
    def __init__(self, num_classes=1):
        super().__init__()
        self.save_hyperparameters()
        # Load FCN ResNet50 pretrained on COCO
        self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
        # Replace the classifier with a new one for binary segmentation.
        # The classifier in FCN has a 'classifier' attribute.
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        # Replicate single channel to three channels since model expects RGB input.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # The model returns a dictionary; we take the 'out' key.
        out = self.model(x)['out']
        # Apply sigmoid activation to get probabilities.
        return torch.sigmoid(out)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        preds = self.forward(images)
        loss = F.binary_cross_entropy(preds, masks)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self.forward(images)
        loss = F.binary_cross_entropy(preds, masks)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
