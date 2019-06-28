import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os
import time
import keras
import importlib
from flask import Flask, render_template, Response ,request,redirect,url_for
import cv2
import tensorflow as tf
from keras import backend as k

graph = tf.get_default_graph()
   
l=[]    
newlabl='.'
        
app = Flask(__name__) 
vc = cv2.VideoCapture(0) 

mod=keras.models.load_model('C:/Users/HP/Desktop/Flask App/model/model-64-64')
flag=0
labels = os.listdir('C:/Users/HP/Desktop/Flask App/data/train/')
print(mod.summary())
print(labels)
p='no one'
temp=0
flag=0
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

@app.route('/',methods=['GET','POST']) 
def index():
	global newlabl 
	if request.method=='POST':
		newlabl=request.form['label']
		createFolder('C:/Users/HP/Desktop/Flask App/data/train/'+newlabl+'/')
		createFolder('C:/Users/HP/Desktop/Flask App/data/test/'+newlabl+'/')
		#vc.release()
		#cv2.destroyAllWindows()
		flag=1
		return redirect(url_for('newlabel'))
	return render_template('index.html',b=temp,i='/static/robot.png',out=p) 
	

def gen():
	temp=0
	k.clear_session()
	graph = tf.get_default_graph()
	sess = tf.Session()
	k.set_session(sess)

	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	
	with graph.as_default():
		mod=keras.models.load_model('C:/Users/HP/Desktop/Flask App/model/model-64-64')

	while True:
		if(flag==1):
			break
		rval, frame = vc.read()

		#cv2.imwrite('C:/Users/ACER/Desktop/web app/static/pic.jpg', frame)
		frame2=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		#print("hii")
		
		
		frame2 = cv2.resize(frame2, (64,64))
		frame2 = img_to_array(frame2)
		frame2 = np.array(frame2, dtype="float32") / 255
		with graph.as_default():
			y_pred = mod.predict(frame2[None,:,:,:]) 

		
		
		l=list(y_pred[0])
		print(l.index(max(l)))
		
		

		p=labels[l.index(max(l))]

		cv2.putText(frame, labels[l.index(max(l))] , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
		cv2.imwrite('C:/Users/HP/Desktop/Flask App/static/pic.jpg', frame)
		



		
		yield (b'--frame\r\n' 
              b'Content-Type: image/jpeg\r\n\r\n' + open('C:/Users/HP/Desktop/Flask App/static/pic.jpg', 'rb').read() + b'\r\n')
	


def lab():
	temp=0
	while True:
		temp=temp+1
		yield str(temp)


@app.route('/label')
def label():
	return Response(lab(),mimetype='text/plane')
	

@app.route('/video_feed') 
def video_feed():
	return Response(gen(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame') 

'''
@app.route('/hello_label')
def hello_label():
	return Response(lab(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')
'''


@app.route('/newlabel')

def newlabel():
	k.clear_session()
	global newlabl 
	return render_template('label.html',label=newlabl)

def input():
	t=0
	while(t<500):
		rval, frame = vc.read()


		#cv2.imwrite('C:/Users/ACER/Desktop/web app/static/pic.jpg', frame)
		#frame2=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		#print("hii")

		'''
		
		frame2 = cv2.resize(frame2, (64,64))
		frame2 = img_to_array(frame2)
		frame2 = np.array(frame2, dtype="float32") / 255
		with graph.as_default():
			y_pred = mod.predict(frame2[None,:,:,:]) 

		
		
		l=list(y_pred[0])
		print(l.index(max(l)))
		
		

		p=labels[l.index(max(l))]


		cv2.putText(frame, labels[l.index(max(l))] , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
		'''


		if(t<400):
			cv2.imwrite('C:/Users/HP/Desktop/Flask App/data/train/'+newlabl+'/pic'+'.'+str(t)+'.jpg', frame)
		else:
			cv2.imwrite('C:/Users/HP/Desktop/Flask App/data/test/'+newlabl+'/pic'+'.'+str(t)+'.jpg', frame)
		
		if(t==499):
			cv2.putText(frame, "capturing completed \n train now.." , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
		else:
			cv2.putText(frame, "capturing frame :"+str(t+1)+"/500" , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

		cv2.imwrite('C:/Users/HP/Desktop/Flask App/static/pic.jpg', frame)

		t+=1

		
		yield (b'--frame\r\n' 
              b'Content-Type: image/jpeg\r\n\r\n' + open('C:/Users/HP/Desktop/Flask App/static/pic.jpg', 'rb').read() + b'\r\n')
	importlib.import_module('network_structure_train')



@app.route('/input_feed') 
def input_feed():
	return Response(input(), mimetype='multipart/x-mixed-replace; boundary=frame') 



if __name__ == '__main__': 
	app.run(host='0.0.0.0', debug=True, threaded=True) 