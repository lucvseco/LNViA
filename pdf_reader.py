import os
import pdfplumber

def load_pdfs_to_txt_with_plumber(directory, output_file):
    with open(output_file, "w", encoding="utf-8") as txt_file:
        for filename in os.listdir(directory):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(directory, filename)
                print(f"Lendo: {filename}")
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    txt_file.write(text)
                    txt_file.write("\n\n")
    print(f"Texto extraído e salvo em: {output_file}")


if __name__ == "__main__":
    pdf_directory = r"C:\Users\vitor\OneDrive\Documentos\LNViA-pdfs_atualizados"
    output_txt_file = "base_textual.txt"
    load_pdfs_to_txt_with_plumber(pdf_directory, output_txt_file)
