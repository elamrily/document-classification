# Documents classification

The goal of this project is to study automatic classification methods for 
scanned documents. Scanned documents have several specificities compared 
to electronic text:

* on can access both the image and the text
* the text is noisy because it is usually produced by OCR
* the layout of the text on the page is complex and is important

These aspects will be studied in this project.

To extract text through the command line, we can pass the following command :

```
python extract_text.py ../Data/train2/xml ../Data/train2/txt/ ../Data/train2/train_text.csv
```

And for the dev2 dataset, we can just replace train2 by dev2 and execute the same command.
