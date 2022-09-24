import cv2
import numpy as np
import glob
import os

menu_options = {
    1: 'Suni Hasar Tespit Modeli',
    2: 'Gerçek Hasar Tespit Modeli',
    3: 'Çıkış',
}

def showOutput(path, model):

    for file in path:
        img = cv2.imread(file)
        img_width = img.shape[1]
        img_height = img.shape[0]

        file_name = os.path.basename(file)
        print(file_name)        

        img_blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB=True, crop=False)
        
        labels = ["Hasarsiz","Hasarli"]
        
        colors = ["169, 1, 0","0, 0, 154"]
        colors = [np.array(color.split(",")).astype("int") for color in colors]
        colors = np.array(colors)
        colors = np.tile(colors, (18,1))
        
        layers = model.getLayerNames()
        
        output_layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()]
        
        model.setInput(img_blob)
        
        detection_layers = model.forward(output_layer)
        
        ids_list = []
        boxes_list = []
        confidence_list = []
        
        for detection_layer in detection_layers:
            for object_detection in detection_layer:
                
                #ilk 5 eleman bounding box ile alakalı değerler
                scores = object_detection[5:]
                predicted_id = np.argmax(scores)
                confidence = scores[predicted_id]
                
                if confidence > 0.30:
                   
                    label = labels[predicted_id]
                    bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
                    (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                    
                    start_x = int(box_center_x - (box_width/2))
                    start_y = int(box_center_y - (box_height/2))
        
                    end_x = start_x + box_width
                    end_y = start_y + box_height
                    
                    ids_list.append(predicted_id)
                    confidence_list.append(float(confidence))
                    boxes_list.append([start_x,start_y,int(box_width),int(box_height)]) 
                    
        max_ids = cv2.dnn.NMSBoxes(boxes_list, confidence_list, 0.5,0.4)
        for max_id in max_ids:
            max_class_id = max_id
            box = boxes_list[max_class_id]
            start_x = box[0]
            start_y = box[1]
            box_width = box[2]
            box_height = box[3]
            
            predicted_id = ids_list[max_class_id]
            label = labels[predicted_id]
            confidence = confidence_list[max_class_id]
            
            
            end_x = start_x + box_width
            end_y = start_y + box_height
                    
            box_color = colors[predicted_id]
            box_color = [int(each) for each in box_color]
        
        
            label = "{}:{:.2f}%".format(label, confidence*100)
            print("Predicted Object {}:".format(label))          
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, 2)
            cv2.putText(img, label, (start_x, start_y-8), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 3)
        
    
        cv2.namedWindow("Damage_Detection",  cv2.WND_PROP_FULLSCREEN)
        #cv2.resizeWindow("Damage_Detection", 640, 480)
        cv2.imshow("Damage_Detection", img)
        cv2.waitKey(0)    
        cv2.destroyAllWindows()

def print_menu():
    for key in menu_options.keys():
        print (key, '--', menu_options[key] )

def option1():
     path = glob.glob("images/suni_hasar/*.png")
     model = cv2.dnn.readNetFromDarknet("scooter-model/yolov4-tiny-custom.cfg","scooter-model/yolov4-tiny-suni_hasar.weights")
     showOutput(path, model)
     
def option2():
    path = glob.glob("images/gercek_hasar/*.png")
    model = cv2.dnn.readNetFromDarknet("scooter-model/yolov4-tiny-custom.cfg","scooter-model/yolov4-tiny-gercek_hasar.weights")
    showOutput(path, model)


if __name__=='__main__':    
    while(True):
        print("\nKullanmak İstediğiniz Modeli Seçiniz")
        print_menu()
        option = ''
        try:
            option = int(input('Seçiminiz: '))
        except:
            print('Lütfen bir sayı giriniz...')
        if option == 1:
           option1()
        elif option == 2:
            option2()
        elif option == 3:
            print('Uygulama Kapatılıyor...')
            exit()
        else:
            print('Lütfen 1 ve 3 arasında bir sayı giriniz.')