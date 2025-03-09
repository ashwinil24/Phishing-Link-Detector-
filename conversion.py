from docx import Document
import pandas as pd
def extract_links_from_docx(doc_path):
    doc = Document(doc_path)  # Open the Word document
    links = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    # Debug: Print extracted links
    print("Extracted Links:", links)  
    return links
# Your exact file path (correctly formatted)
docx_file = r"D:\phishing project\phishing_dataset.csv"
# Extract phishing links
phishing_links = extract_links_from_docx(docx_file)
# Check if any links were extracted
if not phishing_links:
    print("No links found in the document!")
# Save to CSV
df = pd.DataFrame({"url": phishing_links, "label": ["phishing"] * len(phishing_links)})
df.to_csv("phishing_dataset.csv", index=False)
print("Extracted links saved to phishing_dataset.csv")