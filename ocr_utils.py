import easyocr

reader = easyocr.Reader(['en'])

def extract_text(image_path):

    result = reader.readtext(image_path, detail=1)

    filtered_text = []

    for item in result:

        text = item[1]
        confidence = item[2]

        if confidence > 0.4:
            filtered_text.append(text)

    text = " ".join(filtered_text)

    return text