import base64
import os
import csv

SHAPE = 14
# Input and Output Paths
IMAGES_DIR = "images/shape14"
OUTPUT_HTML = rf"shape{SHAPE}_labels.html"
PREDICTIONS_FILE = rf"shape14_tsne_ranking.csv"


# Helper: encode image to base64 for embedding
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# Parse predicted labels
def parse_labels(file_path):
    print("Parsing  labels...")
    images_data = []
    with open(file_path, mode="r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            images_data.append({
                "child_id": row["child_id"],
                "label": row["rank"]
            })
    print(f"Parsed {len(images_data)} labels.")
    return images_data


# Generate HTML with 10 images per row and true labels in parentheses
def generate_html(images_data, images_dir, output_file):
    print("Generating HTML file...")
    # Sort by predicted score
    images_data.sort(key=lambda x: float(x["label"]))

    html = [
        "<html>",
        f"<head><title>Shape {SHAPE} Labels</title></head>",
        "<body>",
        "<h1>Images Sorted by Class</h1>",
        "<div style='display: grid; grid-template-columns: repeat(10, 1fr); grid-gap: 10px;'>"
    ]

    for idx, img in enumerate(images_data, start=1):
        cid = img["child_id"]
        label = img["label"]

        img_path = os.path.join(images_dir, f"{cid}_{SHAPE}mnist.png")
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping.")
            continue
        b64 = encode_image(img_path)

        # Add grid cell
        html.append(
            f"<div style='text-align:center;'>"
            f"<img src='data:image/png;base64,{b64}' alt='{cid}' style='width:100px;height:100px;'><br/>"
            f"<strong>#{idx}</strong><br/>"
            f"{cid.split("_")[0]}<br/>"
            f"{str(label)}"
            f"</div>"
        )
        # After each full row, insert a horizontal rule spanning all columns
        if idx % 10 == 0:
            html.append("<hr style='grid-column:1 / -1; width:100%;' />")

    html += ["</div>", "</body>", "</html>"]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"HTML generated: {output_file}")


# Main execution
if __name__ == "__main__":
    preds = parse_labels(PREDICTIONS_FILE)
    generate_html(preds, IMAGES_DIR, OUTPUT_HTML)
