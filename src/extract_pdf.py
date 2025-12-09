from pypdf import PdfReader

try:
    reader = PdfReader("Atividade1_IA_avancado.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    with open("pdf_content_utf8.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Done writing to pdf_content_utf8.txt")
except Exception as e:
    print(f"Error reading PDF: {e}")
