# 📚 Libris: AI-Augmented Document Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![Ollama](https://img.shields.io/badge/Model-Llama3.2-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Libris**, yerel olarak çalışan (on-premise), gizlilik odaklı bir **Akıllı Doküman Arama ve Özetleme Sistemidir**.

Bu proje, **Bakırçay Üniversitesi BİL440 - YZ Destekli Yazılım Geliştirme** dersi Final Projesi (Project #2) kapsamında geliştirilmiştir.

---

## 🚀 Proje Hakkında

Bu sistem, kullanıcıların PDF, Word (.docx) ve TXT formatındaki dokümanlarını yükleyebildiği ve bu dokümanlar üzerinde Doğal Dil İşleme (NLP) yöntemleriyle soru-cevap yapabildiği bir web uygulamasıdır. 

Proje, **RAG (Retrieval-Augmented Generation)** mimarisini kullanır ve verileri **asla 3. parti bulut sunucularına göndermez**. Tüm işlemler yerel makinede (Localhost) gerçekleşir.

### ✨ Temel Özellikler

* **Çoklu Format Desteği:** PDF, DOCX ve TXT dosyalarını otomatik işleme ve vektörleştirme.
* **Gizlilik Odaklı:** Bulut API'ları (OpenAI vb.) yerine yerel **Ollama (Llama 3.2)** modeli kullanılır.
* **Akıllı Alıntı (Citations):** Verilen cevapların dokümanın hangi parçasından alındığını gösterir.
* **Manuel Tool Calling:** Modelin halüsinasyon görmesini engellemek ve sonsuz döngüleri kırmak için insan tarafından optimize edilmiş özel bir karar mekanizması içerir.
* **Hafıza Yönetimi:** Sohbet geçmişini optimize ederek (token limitine takılmadan) bağlamı korur.

---

## 🛠️ Mimari ve Teknolojiler

Bu proje aşağıdaki teknoloji yığını üzerine inşa edilmiştir:

* **Backend:** Python, Flask
* **LLM Orchestration:** LangChain
* **LLM (Yerel):** Ollama (Llama 3.2)
* **Vector Database:** FAISS (CPU)
* **Embeddings:** HuggingFace (`intfloat/multilingual-e5-large`)
* **Document Parsing:** `pypdf`, `python-docx`

---

## ⚙️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

### Ön Koşullar

1.  **Python 3.10+** yüklü olmalıdır.
2.  **Ollama** bilgisayarınızda kurulu ve çalışıyor olmalıdır.
    * İndirmek için: [ollama.com](https://ollama.com)
    * Modeli çekmek için terminalde: `ollama pull llama3.2`

### Adım Adım Kurulum

1.  **Repoyu Klonlayın:**
    ```bash
    git clone [https://github.com/KULLANICI_ADIN/REPO_ADIN.git](https://github.com/KULLANICI_ADIN/REPO_ADIN.git)
    cd REPO_ADIN
    ```

2.  **Sanal Ortam Oluşturun (Önerilen):**
    ```bash
    python -m venv venv
    # Windows için:
    venv\Scripts\activate
    # Mac/Linux için:
    source venv/bin/activate
    ```

3.  **Gereksinimleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Uygulamayı Başlatın:**
    ```bash
    python upload_rag_chat.py
    ```

5.  **Tarayıcıda Açın:**
    Uygulama başladığında `http://127.0.0.1:5001` adresine gidin.

---

## 🧠 YZ Geliştirme Süreci (AI Development Process)

Bu proje geliştirilirken Yapay Zeka araçları (GitHub Copilot, Claude 3.5, ChatGPT) aktif olarak kullanılmış, ancak kritik mühendislik kararları insan müdahalesiyle yönetilmiştir.

Git commit geçmişimizde aşağıdaki etiketleme standardı kullanılmıştır:

* 🟢 `[AI-generated]`: Temel iskelet ve boilerplate kodlar.
* 🟡 `[AI-assisted]`: YZ önerisiyle yazılan ancak insan tarafından optimize edilen kodlar.
* 🟣 `[Human-written]`: İş mantığı, güvenlik yamaları ve halüsinasyon önleme kuralları.

### Kritik Karar Günlüğü (Decision Log)

| Aşama | Durum | Açıklama |
| :--- | :--- | :--- |
| **Mimari** | 🔴 Reddedildi | YZ'nin önerdiği Cloud Vector DB (Pinecone) veri gizliliği riski nedeniyle reddedildi. Yerel FAISS seçildi. |
| **OCR** | 🟡 Değiştirildi | YZ'nin önerdiği Tesseract OCR çok yavaştı. Yerine Python tabanlı parser'lar (`pypdf`) kullanıldı. |
| **Prompt** | 🟣 İnsan Müdahalesi | Modelin halüsinasyon görmesini engellemek için `HOLY_PROMPT` kural seti sisteme kodlandı. |

---

## 📂 Proje Yapısı

```text
BIL440-Final-Project/
├── upload_rag_chat.py       # Ana uygulama (Flask + RAG Logic)
├── requirements.txt         # Kütüphane bağımlılıkları
├── uploaded_documents/      # Kullanıcının yüklediği geçici dosyalar
├── vector_db_uploaded_faiss/# Oluşturulan vektör veritabanı (FAISS index)
└── README.md                # Proje dokümantasyonu