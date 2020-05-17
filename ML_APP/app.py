from flask import Flask,render_template,url_for,request
import ticket
app = Flask(__name__)

@app.route('/abc',methods=['GET','POST'])
def home():
        my_prediction=""
	if request.method == 'POST':
		comment = request.form['comment']
		my_prediction=ticket.model_predict(comment)
	return render_template('home.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
