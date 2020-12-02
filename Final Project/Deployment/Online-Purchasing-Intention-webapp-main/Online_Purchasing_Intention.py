import streamlit as st
import pickle
import numpy as np 

model = pickle.load(open('final_model.pkl','rb'))

def predict(Administrative,Administrative_Duration,ProductRelated,ExitRates, PageValues):
    input_features = np.array([Administrative,Administrative_Duration,ProductRelated,ExitRates, PageValues]).astype(np.float64).reshape(1,-1)
    prediction = model.predict_proba(input_features)
    pred='{0:.{1}f}'.format(prediction[0][0],4)
    return float(pred)

def main():
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Online Shoppers Purchasing Intention</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    Administrative = st.slider("Administrative",0,27)
    Administrative_Duration = st.number_input("Administrative Duration",0.00)
    ProductRelated = st.slider("Product Related",1,339)
    ExitRates = st.number_input("Exit Rates",0.00,1.00)
    PageValues = st.number_input("Page Values",0.00)
    
    if st.button("Predict"):
        output=predict(Administrative,Administrative_Duration,ProductRelated,ExitRates, PageValues)
        st.success('The probability of making revenue from this customer is {0:.{1}f}%'.format(output * 100,2))


if __name__=='__main__':
    main()
