# GUI設計

## 功能簡述

1. 使用``PyQt5``來進行設計
2. 更改目前的``docx_parser_static.py``之檔案結構，將其變為一個被呼叫的函式庫，並透過GUI介面呼叫，進行parsing的主邏輯及匯出csv
3. 利用``TableView``物件，將parsed_data之內容做為表格顯示到Table上，Table應支援水平和垂直滾輪
4. 在時間軸資料parsing完成後，要支援透過``combo_box``切換Table內顯示的資料類別，每個類別可能有不同的欄位和元素數量
