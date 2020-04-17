# NL2CODE
> Seq2seq model to traduce natural language into Python code.

### Installation

```bash
git clone https://github.com/Jonor127-OP/NL2CODE.git
cd NL2CODE
bash ./getdata.sh  # get datasets
```


**Steps**

- First: run the seq2seq model without attention on CoNaLa.
- Second: run seq2seq model with attention on CoNaLa.
- Third: add a bi-LSTM in the encoder and run it.
- Fourth: Configure decoder to predict action rather than code.

