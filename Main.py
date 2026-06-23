import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
import re
# 換回支持本機地端 Qwen 大模型的 Auto 序列
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalImageTranslator:
    def __init__(self, model_dir="./models"):
        print("正在初始化 EasyOCR...")
        self.ocr = easyocr.Reader(['en'], gpu=True, model_storage_directory=model_dir)
        
        print("正在加載地端 Qwen2.5 行業級大語言翻譯模型 (這可能需要幾分鐘)...")
        #self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype="auto", 
            device_map="auto", 
            cache_dir=model_dir
        )
        
        # 加載中文字體 (Windows 自帶微軟正黑體)
        self.font_path = "C:\\Windows\\Fonts\\msjh.ttc" 
        
        # 💡【核心新增】：建立臨時快取字典，儲存已翻譯的文字，確保一致性並加速
        self.translation_cache = {}
        
    def _detect_design_pack_region(self, img):
        """
        動態偵測設計包圖片（Design Pack Image）的位置
        根據文字框左上角定位：
        - 往左150像素 → 左邊界
        - 往上350像素 → 上邊界
        - 往右350像素 → 右邊界
        - 底部 = 圖片底部
        """
        img_h, img_w = img.shape[:2]
        
        # 策略1: 嘗試通過文字偵測找到 "DESIGN PACK" 或相關標題
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用 EasyOCR 識別文字，找到 DESIGN PACK 的位置
        result = self.ocr.readtext(img, paragraph=False)
        
        design_pack_text = None
        for box, text, confidence in result:
            if "design" in text.lower() and "pack" in text.lower():
                design_pack_text = box
                break
        
        if design_pack_text is not None:
            # 獲取文字框的左上角
            x0, y0 = int(design_pack_text[0][0]), int(design_pack_text[0][1])
            
            # 根據描述計算設計包圖片的範圍
            design_left = max(0, x0 - 150)      # 往左150像素
            design_top = max(0, y0 - 350)       # 往上350像素
            design_right = min(img_w, x0 + 350) # 往右350像素
            design_bottom = img_h               # 圖片底部
            
            return {
                'left': design_left,
                'right': design_right,
                'top': design_top,
                'bottom': design_bottom,
                'anchor_x': x0,
                'anchor_y': y0,
                'method': 'text_detection'
            }
        
        # 策略2: 如果找不到 "DESIGN PACK" 文字，使用輪廓偵測
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # 檢查是否符合設計包圖片的特徵（位於底部，高度約350像素）
            if 250 <= h <= 450 and y + h >= img_h * 0.8:
                # 假設這個輪廓就是設計包圖片
                # 但由於我們的計算是基於文字基準點，這裡用近似值
                design_left = max(0, x - 50)
                design_top = max(0, y - 50)
                design_right = min(img_w, x + w + 50)
                design_bottom = img_h
                
                return {
                    'left': design_left,
                    'right': design_right,
                    'top': design_top,
                    'bottom': design_bottom,
                    'method': 'contour_detection'
                }
        
        # 策略3: 完全無法偵測時，使用預設值（根據常見布局）
        # 假設設計包圖片在右下角區域
        design_top = max(0, img_h - 400)   # 從底部往上400像素
        design_left = max(0, img_w - 600)  # 從右邊往左600像素
        design_right = img_w
        design_bottom = img_h
        
        return {
            'left': design_left,
            'right': design_right,
            'top': design_top,
            'bottom': design_bottom,
            'method': 'default'
        }

    def _is_in_design_pack_region(self, x, y, design_region):
        """
        檢查指定點是否在設計包圖片區域內
        """
        if design_region is None:
            return False
        
        # 簡單的矩形檢查（對於四邊形，使用矩形作為近似）
        return (design_region['left'] <= x <= design_region['right'] and 
                design_region['top'] <= y <= design_region['bottom'])
    
    def _translate_text(self, text):
        """
        地端 AI 智慧翻譯核心：新增雜訊字串攔截，遇到模型拒答直接回傳原值
        """
        cleaned_lookup = text.strip()
        if not cleaned_lookup:
            return ""
            
        # 📌 修正 1：純數字與極短雜訊防禦
        # 如果是純數字直接放行；如果文字長度小於 3 且不是常見的服裝縮寫，直接當成雜訊回傳原值
        if cleaned_lookup.isdigit():
            return cleaned_lookup
        
        if len(cleaned_lookup) < 3 and cleaned_lookup.lower() not in ["in", "cm", "xs", "xl"]:
            return cleaned_lookup

        # 快取命中檢查
        if cleaned_lookup in self.translation_cache:
            return self.translation_cache[cleaned_lookup]

        try:
            # 使用 XML 標籤嚴格控制 Qwen 的輸出格式
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "你是一個精通服裝製衣與技術包（Tech Pack）的專業翻譯官。\n"
                        "你的任務是將輸入的服裝工藝、尺寸、面料、輔料或可能帶有印刷錯誤的英文，精準翻譯為服裝行業通用的繁體中文。\n\n"
                        "【嚴格執行以下鐵律】:\n"
                        "1. 必須使用繁體中文（正體中文）。\n"
                        "2. 英文中原本的數字、型號、尺寸符號（例如 316, 189, 18\"）必須原樣保留，絕對不允許翻譯成中文數字。\n"
                        "3. 遇到拼寫錯誤的英文（EasyOCR 識別錯字），請根據服裝上下文智慧糾錯後再翻譯。\n"
                        "4. 絕對禁止輸出任何括號、註釋、背景解釋、‘好的’、‘術語’等廢話。\n"
                        "5. 請將最終的繁體中文翻譯結果放在 <result> 和 </result> 標籤之間。例如：<result>繁體中文結果</result>"
                    )
                },
                {"role": "user", "content": f"請翻譯以下內容: {cleaned_lookup}"}
            ]
            
            text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
            
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=64, 
                do_sample=False
            )
            
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            raw_translation = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # 同時相容 <result> 和 [result] 的正則提取
            match = re.search(r'[<\[]result[>\]](.*?)[<\[]/result[>\]]', raw_translation, re.DOTALL)
            if match:
                translation = match.group(1).strip()
            else:
                translation = raw_translation
                
            # 終極暴力清洗線
            dirty_words = [
                "<result>", "</result>", "[result]", "[/result]", 
                "result", "術語", "翻譯", ":", "：", "[", "]", "<", ">"
            ]
            for word in dirty_words:
                translation = translation.replace(word, "")
            
            # 二次通用清洗（強行剔除任何括號解釋）
            translation = re.sub(r'（[^）]*）', '', translation)
            translation = re.sub(r'\([^)]*\)', '', translation)
            
            final_translation = translation.strip()

            # 🚨 🌟【核心增強：拒答攔截網】🌟
            # 如果大模型吐回來的中文裡包含了常見的拒答客服關鍵字，代表模型心虛或誤判了
            refusal_keywords = ["抱歉", "無法理解", "提供具體", "問題描述", "信息", "錯誤", "請提供"]
            if any(keyword in final_translation for keyword in refusal_keywords):
                # 遇到這種情況，不要用這段客服廢話，直接判定使用【原英文值】
                final_translation = cleaned_lookup

            # 如果洗完變空值，也返回原詞
            if not final_translation:
                final_translation = cleaned_lookup

            # 寫入快取
            self.translation_cache[cleaned_lookup] = final_translation
            return final_translation
            
        except Exception as e:
            print(f"Qwen 翻譯出錯: {e}")
            return cleaned_lookup

    def translate_and_overlay(self, img_path, output_path="output_translated.png"):
        """
        純 AI 智慧翻譯 + 動態設計包圖片保護 + 框線防禦 + 全局純白背景覆蓋
        """
        print(f"\n--- 開始處理圖片: {img_path} ---")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"無法讀取圖片: {img_path}")
            
        img_h, img_w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 🔍 動態偵測設計包圖片區域
        design_region = self._detect_design_pack_region(img)
        print(f"🔍 偵測到的設計包圖片區域: {design_region}")
        
        # 獲取最精準的單行文字框與坐標
        result = self.ocr.readtext(img_rgb, paragraph=False)
        
        if not result:
            print("未檢測到文本。")
            cv2.imwrite(output_path, img)
            return

        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        for box, text, confidence in result:
            text_strip = text.strip()
            if not text_strip or (len(text_strip) < 2 and not text_strip.isdigit()):
                continue

            x0, y0 = int(box[0][0]), int(box[0][1])
            x2, y2 = int(box[2][0]), int(box[2][1])
            center_x = (x0 + x2) // 2
            center_y = (y0 + y2) // 2
            
            # 檢查是否在設計包圖片區域內
            if self._is_in_design_pack_region(center_x, center_y, design_region):
                print(f"[動態保護] 跳過 '{text_strip}' (位於設計包圖片區)，原圖完好保留。")
                continue

            w, h = x2 - x0, y2 - y0
            if w <= 0 or h <= 0: 
                continue

            # 框線物理防禦：往內縮進 1~2 像素，防止切斷表格黑線
            pad = 2 if h > 15 else 1
            erase_x0 = max(0, x0 + pad)
            erase_y0 = max(0, y0 + pad)
            erase_x2 = max(erase_x0 + 1, x2 - pad)
            erase_y2 = max(erase_y0 + 1, y2 - pad)

            # 強制使用 RGB 純白色進行原地覆蓋
            draw.rectangle([erase_x0, erase_y0, erase_x2, erase_y2], fill=(255, 255, 255))
            
            # 呼叫帶有快取機制的翻譯核心
            translated_zh = self._translate_text(text_strip)
            print(f"[替換] {text_strip}  ==>  {translated_zh}")
            
            # 統一文字大小（可調整此數值改變所有文字的大小）
            FIXED_FONT_SIZE = 12  # 您可以調整這個數字，建議範圍 12-18

            try:
                font = ImageFont.truetype(self.font_path, FIXED_FONT_SIZE)
                # 檢查文字是否會超出框線，如果超出則稍微縮小
                left, top, right, bottom = draw.textbbox((0, 0), translated_zh, font=font)
                text_width = right - left
                text_height = bottom - top
                
                # 如果文字寬度超出框的寬度，嘗試縮小字體
                current_size = FIXED_FONT_SIZE
                while text_width > int(w * 0.92) and current_size > 10:
                    current_size -= 1
                    font = ImageFont.truetype(self.font_path, current_size)
                    left, top, right, bottom = draw.textbbox((0, 0), translated_zh, font=font)
                    text_width = right - left
            except:
                font = ImageFont.load_default()

            # 計算文字置中位置（可選）
            text_x = x0 + (w - text_width) // 2
            text_y = y0 + (h - text_height) // 2

            # 將繁體中文寫入純白色的背景格子上
            draw.text((text_x, text_y), translated_zh, fill=(0, 0, 0), font=font)

        final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, final_img)
        print(f"🎉 處理完成！已啟用快取記憶字典，繁體圖片已保存至: {output_path}")

# --- 執行入口 ---
if __name__ == "__main__":
    test_img_path = "input/techpack_img 1.png"
    output_img_path = "output/translated_techpack.png"
    
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    if os.path.exists(test_img_path):
        translator = LocalImageTranslator()
        translator.translate_and_overlay(test_img_path, output_img_path)
    else:
        print(f"請確保測試圖片存在於: {test_img_path}")
