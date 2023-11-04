from SuperDataLoader import *


class CityscapesLoader(SuperDataLoader):


    def __init__(self, **kwargs):

        super().__init__(**kwargs)


        self.datasetRootPath = Path(kwargs["cityscapesRootPath"])
        
        tmpDataset = []
        for imgPath in glob.glob(str(Path.joinpath(self.datasetRootPath, "images", self.mode, "*.jpg"))):
            
            if (exists(str(imgPath).replace("images","color"))) and (exists(str(imgPath).replace("images","masks"))):
                tmpDataset.append(imgPath)
        
        self.dataset = np.array(tmpDataset)  
        self.labels = {}
        Tmplabels = {label.name.replace(" ", "-"):label.color for label in city_labels}
        for mapillaryNewColor in self.mapillaryNewColors:
            for cityscapesLabel in Tmplabels:
                if cityscapesLabel in mapillaryNewColor:
                    self.labels[mapillaryNewColor] = Tmplabels[cityscapesLabel]


        
        self.pixel_to_color = np.vectorize(self.return_color)



        self.transform_in = preprocess_in = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2867, 0.3250, 0.2837], [0.1862, 0.1895, 0.1865]),
    transforms.Resize(self.imgSize)
])
        self.transform_ou = preprocess_out = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(self.imgSize)
])

    def get_num_classes(self):
        if self.reducedCategories:
            return len(self.reducedCategoriesColors)
        else:
            return len(self.labels)
        
    def __len__(self):
        return len(self.dataset)



    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.dataset[idx])
        seg_mask = np.array(Image.open(self.dataset[idx].replace("images", "masks")))
        seg_color = np.array(Image.open(self.dataset[idx].replace("images", "color")).convert('RGB'))
        label, seg_color = self.create_prob_mask(seg_mask, seg_color)

        if self.transform_in:
            img = self.transform_in(img)
            seg_color = transforms.ToTensor()(seg_color)
        if self.transform_ou:
            label = self.transform_ou(label)

        return {'image': img, 'label': label, "seg": seg_color}
