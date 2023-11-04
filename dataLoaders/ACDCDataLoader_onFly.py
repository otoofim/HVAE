from SuperDataLoader import *




def prepareAugFuncs(imgSize):


    return {

        "org":T.Compose([
                T.ToTensor(),
                T.Resize(imgSize),
                T.ToPILImage(),
            ],
            ),    


#         "crop":T.Compose([
#                 T.ToTensor(),
#                 T.CenterCrop(size = 500),
#                 T.Resize(imgSize),
#                 T.ToPILImage(),
#             ],
#             ),        


#         "grayScale":T.Compose([
#             T.ToTensor(),
#             T.Grayscale(num_output_channels=3),
#             T.Resize(imgSize),
#             T.ToPILImage(),
#             ],
#             ),


#         "colorJitter":T.Compose([
#             T.ToTensor(),
#             T.ColorJitter(brightness = (0.5,1), contrast = (0.5,1), saturation = (0.5,1), hue = 0.5),
#             T.Resize(imgSize),
#             T.ToPILImage(),
#             ],
#             ),



#         "gaussianBlur":T.Compose([
#             T.ToTensor(),
#             T.GaussianBlur(kernel_size = (51, 51), sigma = (5, 5)),
#             T.Resize(imgSize),
#             T.ToPILImage(),
#             ],
#             ),



#         "rotation":T.Compose([
#             T.ToTensor(),
#             T.RandomRotation(degrees = (-30, 30)),
#             T.Resize(imgSize),
#             T.ToPILImage(),
#             ],
#             ),


#         "elastic":T.Compose([
#             T.ToTensor(),
#             T.ElasticTransform(alpha = 500.),
#             T.Resize(imgSize),
#             T.ToPILImage(),
#             ],
#             ),




#         "invert":T.Compose([
#             T.ToTensor(),
#             T.RandomInvert(p = 1.),
#             T.Resize(imgSize),
#             T.ToPILImage(),
#             ],
#             ),



#         "solarize":T.Compose([
#             T.ToTensor(),
#             T.RandomSolarize(threshold = 0.05, p = 1.),
#             T.Resize(imgSize),
#             T.ToPILImage(),
#             ],
#             ),



#         "augMix":T.Compose([

#             T.AugMix(severity = 10, mixture_width = 10),
#             T.ToTensor(),
#             T.Resize(imgSize),
#             T.ToPILImage(),
#             ],
#             ),


#         "posterize":T.Compose([

#             T.RandomPosterize(bits = 2, p = 1.),
#             T.ToTensor(),
#             T.Resize(imgSize),
#             T.ToPILImage(),
#             ],
#             ),


#         "erasing":T.Compose([
#             T.ToTensor(),
#             T.RandomErasing(p=1., scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
#             T.Resize(imgSize),
#             T.ToPILImage(),
#             ],
#             ),
    }





class ACDC_onFly(Dataset):
    
    def __init__(self, **kwargs):
        
        super().__init__()
        
        self.datasetRootPath = kwargs["ACDCRootPath"]
        self.datasetRootPath = kwargs["ACDCRootPath"]
        self.mode = kwargs["mode"]
        self.imgSize = kwargs["input_img_dim"]
        
        self.imgNamesList = glob.glob(join(self.datasetRootPath, "*", kwargs["mode"], "*", "*"))
        
        self.augmenters = prepareAugFuncs(self.imgSize)
        
        
        self.transform_in = preprocess_in = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize(self.imgSize)
        ])
        self.transform_ou = preprocess_out = transforms.Compose([
            transforms.Resize(self.imgSize)
        ])
    
    
    def __len__(self):
#         return 40
        return len(glob.glob(join(self.datasetRootPath, "*", self.mode, "*", "*")))*len(self.augmenters.keys())
    
    
    def get_num_classes(self):
        return len(city_labels)

    def __getitem__(self, idx):
        
        
        org_img, ref_img, seg_mask, seg_color = self.generateSample(idx)
               
        
        if self.transform_in:
            org_img = self.transform_in(org_img)
            ref_img = self.transform_in(ref_img)
            seg_color = transforms.Resize(self.imgSize)(transforms.ToTensor()(seg_color))
        if self.transform_ou:
            label = torch.as_tensor(np.array(self.transform_ou(seg_mask)), dtype = torch.int64)
            label = F.one_hot(label, num_classes=self.get_num_classes()).permute(2,0,1)*1.
        
        
        return {'image': [org_img, ref_img], 'label': label, "seg": seg_color}
    
    
    def generateSample(self, idx):
        
        
        org_img_idx = idx//len(self.augmenters.keys())
        org_img = Image.open(self.imgNamesList[org_img_idx])
        ref_img = Image.open(self.imgNamesList[org_img_idx].replace(self.mode,f"{self.mode}_ref").replace("_rgb", "_rgb_ref"))
        seg_mask = Image.open(self.imgNamesList[org_img_idx].replace("images","gt_uncertainty").replace("/rgb_anon","/gt").replace("_rgb_anon","_gt_labelIds"))
        seg_color = Image.open(self.imgNamesList[org_img_idx].replace("images","gt_uncertainty").replace("/rgb_anon","/gt").replace("_rgb_anon","_gt_labelColor"))
        
        
        seed = np.random.randint(27)
        
        augKey = list(self.augmenters.keys())[idx%len(self.augmenters.keys())]
        
        augmenterFunc = self.augmenters[augKey]
        
        if augKey == "crop" or augKey == "rotation":


            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            org_img = augmenterFunc(org_img)
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            ref_img = augmenterFunc(ref_img)
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            seg_color = augmenterFunc(seg_color)
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            seg_mask = augmenterFunc(seg_mask)
            

        elif augKey == "erasing" or augKey == "augMix":
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            org_img = augmenterFunc(org_img)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            ref_img = augmenterFunc(ref_img)
            
        else:

            org_img = augmenterFunc(org_img)
            ref_img = augmenterFunc(ref_img)
        
        return org_img, ref_img, seg_mask, seg_color
        
        
        
    def prMask_to_color(self, segs):
        
        vec_func = np.vectorize(self.myColour2rgb, otypes=[object])

#         outMap = (np.stack(vec_func([sample for sample in segs]), axis = 0)*256).astype(np.uint8)
        outMap = np.stack(vec_func([sample for sample in segs]), axis = 0)

        img = torch.from_numpy(outMap)/256.
        return img.permute(0,3,1,2)
    
    
    
    def myColour2rgb(self, sample):
        
        
        cityscapesColour = {label.id:label.color for label in city_labels}
        colours = list(collections.OrderedDict(sorted(cityscapesColour.items())).values())
        colours = np.array(colours[2:])

        seg = torch.argmax(sample, dim = 0).numpy()
        colourMap = color.label2rgb(seg, colors = colours, bg_label=0, image_alpha = 1., alpha = 1.)
        return colourMap
