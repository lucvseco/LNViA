import os
from PyPDF2 import PdfReader

# Função para carregar e extrair texto de todos os PDFs de um diretório
def load_pdfs_to_txt(directory, output_file):
    with open(output_file, "w", encoding="utf-8") as txt_file:
        for filename in os.listdir(directory):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(directory, filename)
                print(f"Lendo: {filename}")
                with open(file_path, "rb") as f:
                    reader = PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    txt_file.write(text)
                    txt_file.write("\n\n")  # Separar conteúdos de diferentes PDFs
    print(f"Texto extraído e salvo em: {output_file}")

if __name__ == "__main__":
    # Diretório com os PDFs
    pdf_directory = r"C:\Users\vitor\Downloads\drive-download-20250209T155226Z-001"  #TODO: Alterar para o caminho da sua máquina
    # Caminho para o arquivo de saída
    output_txt_file = "base_textual.txt"
    load_pdfs_to_txt(pdf_directory, output_txt_file)
