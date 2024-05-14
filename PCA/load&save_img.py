import os
from PIL import Image
import numpy as np
import pandas as pd
from natsort import natsorted

class ImageProcessor:
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
    def load_images(self):
        all_files = []
        for class_dir in natsorted(os.listdir(self.main_dir)):
            class_dir_path = os.path.join(self.main_dir, class_dir)
            if os.path.isdir(class_dir_path):
                sorted_files = natsorted(filter(lambda x: x.endswith('.bmp'), os.listdir(class_dir_path)))
                all_files.extend([(class_dir, f) for f in sorted_files])
        return all_files
    
    def process_images(self, all_files):
        last_class_dir = None
        for i, (class_dir, filename) in enumerate(all_files):
            
            if class_dir != last_class_dir:  # 如果當前的 class_dir 與上一個不同，則印出
                print(f"======================Class directory: {class_dir}===========================")
                last_class_dir = class_dir  # 更新 last_class_dir
            try:
                # 從檔案名稱中解析出最後的數字
                number = int(filename.split('.')[0]) 
                print(f"Filename: {filename}")   

                # 開啟圖像檔案
                img = Image.open(os.path.join(self.main_dir, class_dir, filename))
                img_array = np.array(img).reshape(-1)
                if number % 2 == 0:  # 偶數索引
                    self.test_images.append(img_array)
                    self.test_labels.append(class_dir)
                    print(f"Loaded image file: {filename} from directory: {class_dir} into test_images")
                else:  # 奇數索引
                    self.train_images.append(img_array)
                    self.train_labels.append(class_dir)
                    print(f"Loaded image file: {filename} from directory: {class_dir} into train_images")
            except IOError:
                print(f"Error opening image file: {filename}")
        print("=" * 30 + "Done" + "=" * 30)

        # 轉換為 NumPy 陣列
        self.train_images = np.array(self.train_images)
        self.train_labels = np.array(self.train_labels)
        self.test_images = np.array(self.test_images)
        self.test_labels = np.array(self.test_labels)


def check_numpy_array(file, file_name):
    print(f"Checking file: {file_name}")
    print(f"Number of images: {len(file)}")
    print(f"Shape of images: {file.shape}")

def main():
    # 主目錄的路徑
    main_dir = 'D:\\GitHub\\courseML\\PCA\\ORL3232'

    # 創建 ImageProcessor 類別的實例
    processor = ImageProcessor(main_dir)
    all_files = processor.load_images()
    processor.process_images(all_files)

    # 獲取數據
    train_images = processor.train_images
    train_labels = processor.train_labels
    test_images = processor.test_images
    test_labels = processor.test_labels

    # 檢查數據
    check_numpy_array(train_images, "train_images")
    check_numpy_array(train_labels, "train_labels")
    check_numpy_array(test_images, "test_images")
    check_numpy_array(test_labels, "test_labels")

        
    # 轉換為 DataFrame
    train_images_df = pd.DataFrame(train_images)
    train_labels_df = pd.DataFrame(train_labels, columns=['label'])
    test_images_df = pd.DataFrame(test_images)
    test_labels_df = pd.DataFrame(test_labels, columns=['label'])

    # 保存為 CSV (index=False不要保存row索引；header=False不要保存column名)
    train_images_df.to_csv(r'D:\GitHub\courseML\PCA\ORL_dataset\train_images.csv', index=False,  header=False)
    train_labels_df.to_csv(r'D:\GitHub\courseML\PCA\ORL_dataset\train_labels.csv', index=False,  header=False)
    test_images_df.to_csv(r'D:\GitHub\courseML\PCA\ORL_dataset\test_images.csv', index=False,  header=False)
    test_labels_df.to_csv(r'D:\GitHub\courseML\PCA\ORL_dataset\test_labels.csv', index=False,  header=False)

if __name__ == "__main__":
    main()
