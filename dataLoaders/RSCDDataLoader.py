from SuperDataLoader import *


class RSCDDataLoader(SuperDataLoader):


    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        
        self.RSCD_cat = kwargs["RSCD_cat"]
        self.datasetRootPath = Path(kwargs["RSCDRootPath"])
        self.RSCDClassNames = RSCDClassNames
        
        tmpDataset = []
        
        if self.mode == "val":
            tmpDataset.extend([path for path in glob.glob(str(Path.joinpath(self.datasetRootPath, self.mode, "*.jpg"))) if any([cat.replace("_","-") in "-".join(path.split(os.sep)[-1].split(".")[0].split("-")[1:]).replace("_","-") for cat in self.RSCD_cat])])
            
            
        elif self.mode == "train":
            for cat in self.RSCD_cat:
                tmpDataset.extend(glob.glob(str(Path.joinpath(self.datasetRootPath, self.mode, cat, "*.jpg"))))
        
        tmpDataset = np.array(tmpDataset)
        self.dataset = tmpDataset
#         if self.mode == "train":
#             self.dataset = tmpDataset[:(len(tmpDataset)*0.8)]
#         else:
#             self.dataset = tmpDataset[(len(tmpDataset)*0.8):]

        
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

        img = Image.open(self.dataset[idx])
#         print(self.dataset[idx].split(os.sep)[-2])
#         print(self.dataset[idx])
    
#         cat = self.RSCDClassNames[self.dataset[idx].split(os.sep)[-2]]
        cat = self.RSCDClassNames["-".join(self.dataset[idx].split(os.sep)[-1].split(".")[0].split("-")[1:])]
        seg_mask = np.full_like(np.array(img), list(self.reducedCategoriesColors.keys()).index(cat))
        label, seg_color, fricLabel = self.create_prob_mask_patches(seg_mask[:,:,0])

        if self.transform_in:
            img = self.transform_in(img)
            seg_color = transforms.Resize((256,256))(transforms.ToTensor()(seg_color))
        if self.transform_ou:
            label = self.transform_ou(label)
            fricLabel = self.transform_ou(fricLabel)

        return {'image': img.type(torch.float), 'label': label.type(torch.float), "seg": seg_color.type(torch.float), "FriLabel": fricLabel.type(torch.float)}

