from pathlib import Path

from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from imageScanner.forms import UserImage

from google.oauth2 import service_account
from google.cloud import vision
from openai.embeddings_utils import cosine_similarity
import io
import openai
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def analyze_image(image_path, client_creds, feature_types):
    credentials = service_account.Credentials.from_service_account_file(
        filename=client_creds,
        scopes=["https://www.googleapis.com/auth/cloud-platform"])
    client = vision.ImageAnnotatorClient(credentials=credentials)
    # load the input image as a raw binary file (this file will be
    # submitted to the Google Cloud Vision API)
    with io.open(str(Path(__file__).resolve().parent.parent) + str(image_path.url), "rb") as f:
        byteImage = f.read()

    image = vision.Image(content=byteImage)
    features = [vision.Feature(type_=feature_type) for feature_type in feature_types]
    request = vision.AnnotateImageRequest(image=image, features=features)

    response = client.annotate_image(request=request)
    # check to see if there was an error when making a request to the API
    if response.error.message:
        raise Exception(
            "{}\nFor more info on errors, check:\n"
            "https://cloud.google.com/apis/design/errors".format(
                response.error.message))
    return response


def labels_df(response: vision.AnnotateImageResponse):
    tags = ''
    for label in response.label_annotations:
        tags = tags + " " + str(label.description)
    return tags


def text_df(response: vision.AnnotateImageResponse):
    tags = ''
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    tags = tags + " " + str(word_text)
    return tags


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


def search_reviews(df, product_description, n=5, pprint=True):
    embedding = get_embedding(product_description, model='text-embedding-ada-002')
    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    return res['similarities'].mean()


def image_request(request):
    if request.method == 'POST':
        form = UserImage(request.POST, request.FILES)
        if form.is_valid():
            form = form.save()
            render_to_string('image.form.html', {'message': 'Please wait while we are checking the image!'})
            img_object = form.image

            features = [vision.Feature.Type.LABEL_DETECTION, vision.Feature.Type.DOCUMENT_TEXT_DETECTION]

            client_creds = 'client_creds.json'

            response = analyze_image(img_object, client_creds, features)
            str_1 = labels_df(response)
            str_2 = text_df(response)
            str = str_1 + str_2
            user_complaint_prose = form.user_complaint_prose
            X_list = word_tokenize(str.lower())
            Y_list = word_tokenize(user_complaint_prose.lower())

            # sw contains the list of stopwords
            sw = stopwords.words('english')
            l1 = []
            l2 = []

            # remove stop words from the string
            X_set = {w for w in X_list if not w in sw}
            Y_set = {w for w in Y_list if not w in sw}

            # form a set containing keywords of both strings
            rvector = X_set.union(Y_set)
            for w in rvector:
                if w in X_set:
                    l1.append(1)  # create a vector
                else:
                    l1.append(0)
                if w in Y_set:
                    l2.append(1)
                else:
                    l2.append(0)
            c = 0

            # cosine formula
            for i in range(len(rvector)):
                c += l1[i] * l2[i]
            cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
            score = cosine
            if score >= 0.05:
                result = "User complaint prose matches the image attatched"
            else:
                result = "User complaint prose does not match the image attatched, need more precised description!"

            return render(request, 'image.form.html', {'form': form, 'img_obj': img_object, 'result': result})
    else:
        form = UserImage()

    return render(request, 'image.form.html', {'form': form})
