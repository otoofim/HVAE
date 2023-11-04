from SuperDataLoader import *

class mapillaryDataLoader(SuperDataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        #setting variables
        self.datasetRootPath = Path(kwargs["mapillaryRootPath"])
        
        
        #
        tmpDataset = []
        for imgPath in glob.glob(str(Path.joinpath(self.datasetRootPath, "images", kwargs["mode"], "*.png"))):
            if (exists(str(imgPath).replace("images","color"))) and (exists(str(imgPath).replace("images","masks"))):
                tmpDataset.append(imgPath)
        self.dataset = np.random.choice(np.array(tmpDataset), int(len(tmpDataset)*kwargs["MapillSubsample"]), replace = False)
        
        labels = {}
        with open(str(Path.joinpath(self.datasetRootPath, "config_v2.0.json"))) as jsonfile:
            config = json.load(jsonfile)
            for label in config['labels']:
                labels[label['name']] = label['color']
        self.labels = labels
        self.NewColors = new_labels
                
        self.transform_in = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4191, 0.4586, 0.4700], [0.2553, 0.2675, 0.2945]),
            transforms.Resize(self.imgSize)
        ])
        
        self.transform_ou = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.imgSize)
        ])

        
    def get_num_classes(self):
        if self.reducedCategories:
            return len(self.reducedCategoriesColors)
        else:
            return len(self.labels)
        
    def __len__(self):
        return len(self.dataset[:])

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        img = Image.open(self.dataset[idx])

        seg_mask = np.array(Image.open(self.dataset[idx].replace("images", "masks")))
        seg_color = np.array(Image.open(self.dataset[idx].replace("images", "color")).convert('RGB'))
        
        label, seg_color = self.create_prob_mask(seg_mask, seg_color)

        if self.transform_in:
            img = self.transform_in(img)
            seg_color = transforms.Resize(self.imgSize)(transforms.ToTensor()(seg_color))
        if self.transform_ou:
            label = self.transform_ou(label)

        return {'image': img.type(torch.float), 'label': label.type(torch.float), "seg": seg_color.type(torch.float)}

