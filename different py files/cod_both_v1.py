class FCNResNet50MultiTask(pl.LightningModule):
    def __init__(self, num_classes=3):
        super().__init__()
        self.save_hyperparameters()

        # If you want to load your segmentation-only checkpoint,
        # you can call self.load_state_dict(torch.load('path_to_checkpoint.ckpt')['state_dict'], strict=False)

        self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
        # Replace the segmentation classifier head with one for binary segmentation
        self.model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)

        # classification branch - uses backbone features
        # important - backbone of FCN ResNet50 outputs features with 2048 channels
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 128),  # changed to 2048 input features
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Ensure input is 3-channel (replicate grayscale if needed)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Segmentation branch
        seg_out = self.model(x)['out']
        seg_out = torch.sigmoid(seg_out)

        # Classification branch: Extract features from backbone.
        features = self.model.backbone(x)['out']  # Expected shape: [B, 2048, H, W]
        class_out = self.classifier(features)

        return seg_out, class_out

    def freeze_decoder(self):
        # Freeze the segmentation head (the FCN classifier) so that segmentation weights remain untouched.
        for param in self.model.classifier.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        # Batch now contains (images, masks, labels)
        images, masks, labels = batch
        seg_out, class_out = self.forward(images)
        seg_loss = F.binary_cross_entropy(seg_out, masks)
        class_loss = F.cross_entropy(class_out, labels)
        total_loss = seg_loss + class_loss
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_seg_loss', seg_loss, prog_bar=True)
        self.log('train_class_loss', class_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, masks, labels = batch
        seg_out, class_out = self.forward(images)
        seg_loss = F.binary_cross_entropy(seg_out, masks)
        class_loss = F.cross_entropy(class_out, labels)
        total_loss = seg_loss + class_loss
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_seg_loss', seg_loss, prog_bar=True)
        self.log('val_class_loss', class_loss, prog_bar=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        images, masks, labels = batch
        seg_out, class_out = self.forward(images)
        seg_loss = F.binary_cross_entropy(seg_out, masks)
        class_loss = F.cross_entropy(class_out, labels)
        total_loss = seg_loss + class_loss
        self.log('test_loss', total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
