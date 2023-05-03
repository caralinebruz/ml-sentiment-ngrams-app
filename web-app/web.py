import torch
from main import text_pipeline
from flask import Flask, request, render_template
from main import TextClassificationModel

app = Flask(__name__)

# location of the model file 
model_filename = 'ngram.pth'
MODEL_DIR = os.environ.get("MODEL_DIR")
model_fullpath = MODEL_DIR + '/' + model_filename
torch.save(model.state_dict(), model_fullpath)

model = torch.load('./ngram.pth')
model.eval()

ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    userinput = request.form['userinput']
    with torch.no_grad():
        text = torch.tensor(text_pipeline(userinput))
        output = model(text, torch.tensor([0]))
        prediction = ag_news_label[output.argmax(1).item() + 1]
    return render_template('index.html', predicted_value = f'{prediction}')

if __name__ == '__main__':
    app.run(debug=True)



# ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
#     enduring the season’s worst weather conditions on Sunday at The \
#     Open on his way to a closing 75 at Royal Portrush, which \
#     considering the wind and the rain was a respectable showing. \
#     Thursday’s first round at the WGC-FedEx St. Jude Invitational \
#     was another story. With temperatures in the mid-80s and hardly any \
#     wind, the Spaniard was 13 strokes better in a flawless round. \
#     Thanks to his best putting performance on the PGA Tour, Rahm \
#     finished with an 8-under 62 for a three-stroke lead, which \
#     was even more impressive considering he’d never played the \
#     front nine at TPC Southwind."