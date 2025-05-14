import cv2
import matplotlib.pyplot as mplt
import numpy as np


#Renk dağılımı (Histogram) çıkarma
def Create_Histogram(folder_name,images):
    for image_name, image in images.items():
        
        histograms = cv2.calcHist(image,[0],None,[255],[0,256])
        mplt.plot(histograms)
        mplt.title("Histogram Of "+image_name.replace(".tif",""))
        mplt.xlabel("Value Of Grey Tone")
        mplt.ylabel("Value Of Pixels")
        mplt.savefig(folder_name+"/"+image_name.replace(".tif",".png"), dpi=100,bbox_inches='tight' )
        mplt.close()

# Resimleri ikilileştirme (Binarization)
def Binarization(folder_name,images):
    count = 0
    for image_name, image in images.items():
        if count < 2:
            th_Value = 150
        else:
            th_Value = 50
        ret, thresh = cv2.threshold(image, th_Value, 255, cv2.THRESH_BINARY)
        if folder_name == "binarizationOutputs":
            bin_images[image_name]= thresh
            mplt.imshow(bin_images[image_name], cmap='gray')
        elif folder_name == "ContrastAdjustedBinarizationOutputs":
            cont_adj_bin_images[image_name]= thresh
            mplt.imshow(cont_adj_bin_images[image_name], cmap='gray')
        mplt.axis("off")
        mplt.title(image_name.replace(".tif",".png"))
        count += 1
        mplt.savefig(folder_name+"/"+image_name.replace(".tif",".png"), dpi=100,bbox_inches='tight')
        

def Zoning(folder_name,images):
    threshold_values = [50,125,190]
    colors = [(255, 0, 120), (255, 0, 0), (0, 120, 255), (120, 255, 0)]
    for image_name, image in images.items():
        new_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8) 
        new_image[image <= threshold_values[0]] = colors[0]
        new_image[(image > threshold_values[0]) & (image <= threshold_values[1])] = colors[1]
        new_image[(image > threshold_values[1]) & (image <= threshold_values[2])] = colors[2]
        new_image[image > threshold_values[2]] = colors[3]
        mplt.axis("off")
        mplt.title(image_name.replace(".tif",".png"))
        if folder_name == "ZoningOutputs":
            zoned_images[image_name]= new_image
        elif folder_name == "ContrastAdjustedZoningOutputs":
            cont_adj_zoned_images[image_name]= new_image
        zoned_images[image_name]= new_image
        mplt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        mplt.savefig(folder_name+"/"+image_name.replace(".tif",".png"), dpi=100,bbox_inches='tight')
        mplt.clf()

def Morp_Process(folder_name,images):
    for image_name,image in images.items(): 
        kernel = np.ones((3, 3), np.uint8)
        #EROSION İŞLEMİ
        erosion = cv2.erode(image,kernel,iterations=2)
        mplt.imshow(erosion, cmap='gray')
        mplt.axis("off")
        mplt.title(image_name.replace(".tif",".png"))
        mplt.savefig(folder_name+"/"+"erosion_"+image_name.replace(".tif",".png"), dpi=100,bbox_inches='tight')
        #DILATION İŞLEMİ
        dilation =cv2.dilate(image,kernel,iterations=2)
        mplt.imshow(dilation, cmap='gray')
        mplt.axis("off")
        mplt.title(image_name.replace(".tif",".png"))
        mplt.savefig(folder_name+"/"+"dilation_"+image_name.replace(".tif",".png"), dpi=100,bbox_inches='tight')
        #CLOSING İŞLEMİ
        closing =cv2.erode(dilation,kernel,iterations=2)
        mplt.imshow(closing, cmap='gray')
        mplt.axis("off")
        mplt.title(image_name.replace(".tif",".png"))
        mplt.savefig(folder_name+"/"+"closing_"+image_name.replace(".tif",".png"), dpi=100,bbox_inches='tight')
        #OPENING İŞLEMİ
        opening =cv2.dilate(erosion,kernel,iterations=2)
        mplt.imshow(opening, cmap='gray')
        mplt.axis("off")
        mplt.title(image_name.replace(".tif",".png"))
        mplt.savefig(folder_name+"/"+"opening_"+image_name.replace(".tif",".png"), dpi=100,bbox_inches='tight')

def Region_Growing(folder_name,images,seed,t_hold):
    for image_name,image in images.items():
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x,y = seed
        new_img = np.zeros_like(grayscale_image)
        new_img[seed] = 255
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        stack  =[(x, y)]

        while stack:
            new_x,new_y = stack.pop()

            for move_direction in movements:
                m_x, m_y = new_x + move_direction[0], new_y + move_direction[1]
                if 0 <= m_x <  grayscale_image.shape[0] and 0 <= m_y <  grayscale_image.shape[1]:
                     if np.array_equal(new_img[m_x, m_y], 0) and abs(int(grayscale_image[m_x, m_y]) - int(grayscale_image[new_x, new_y])) < t_hold:
                        new_img[m_x, m_y] = 1
                        stack.append((m_x, m_y))
        mplt.imshow(new_img, cmap='gray')
        mplt.axis("off")
        mplt.title(image_name.replace(".tif",".png"))
        mplt.savefig(folder_name+"/"+"rgrowing_"+image_name.replace(".tif",".png"), dpi=100,bbox_inches='tight')


     
cont_adj_zoned_images = {}
zoned_images = {}
bin_images = {}
cont_adj_bin_images = {}
equalized_images={}
contrast_adjusted_images = {}
seed = (100,150)
image_name = ["headCT.tif","chest.tif","breast.tif","fetus.tif"]
images = {}
for image in image_name:    
    image_path = "Images/" + image
    images[image] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    alpha = 1.15  # Kontrast faktörü (1.0, orijinal kontrast)
    beta = 15    # Parlaklık (0-100 arası)
    contrast_adjusted_image = cv2.convertScaleAbs(images[image], alpha=alpha, beta=beta)
    contrast_adjusted_images[image] = contrast_adjusted_image
    contrast_adjusted_image_path = "ContrastAdjustedImages/" + image
    cv2.imwrite(contrast_adjusted_image_path, contrast_adjusted_image)


        
print("Images downloaded succesfully!")
Create_Histogram("histogramOutputs",images)
Create_Histogram("ContrastAdjustedHistogramOutputs",contrast_adjusted_images)
Binarization("binarizationOutputs",images)
Binarization("ContrastAdjustedBinarizationOutputs",contrast_adjusted_images)
Zoning("ZoningOutputs",images)
Zoning("ContrastAdjustedZoningOutputs",contrast_adjusted_images)
Morp_Process("ContrastAdjustedMorpOutputs",cont_adj_bin_images)
Morp_Process("MorpOutputs",bin_images)
Region_Growing("R_GrowingOutputs",zoned_images,seed,1)
Region_Growing("ContrastAdjusted_R_GrowingOutputs",cont_adj_zoned_images,seed,1)
print("Images processing complete succesfully!")







    
