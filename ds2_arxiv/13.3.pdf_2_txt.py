import glob, os

def main():
    os.makedirs("data/pdf_txt_4422/", exist_ok=True)
    pdfs = glob.glob("data/pdf_4422/*.pdf")
    for pdf in pdfs:
        arxiv_id = pdf.split("/")[-1][:-4]
        print(arxiv_id)
        txt = f"data/pdf_txt_4422/{arxiv_id}.txt"
        os.system(f"pdftotext -l 1 -layout {pdf} {txt}")

if __name__=="__main__":
    main()