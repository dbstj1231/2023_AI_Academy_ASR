# 2023_AI_Academy_ASR 
Notebooks for "AI 응용-음성"

---------------
# Course Description
## 0. Tutorial
Tutorial for python and data science packages
- python review
- numpy
- matplotlib
- PyTorch Tensor
  
## 1. Audio file handling
Audio file handling using torchaudio
- Load audio file(torchaudio.load)
- Feature extraction(Mel-spectrogram, MFCC)
  
## 3. Audio Classification using MLP
Audio MNIST classification using MLP(torch.Linear)

## 4. CTC
Simple Exercise(model training using CTC loss) for Connectionist Temporal Classification

## 5. Whisper
Exercise using OpenAI - Whisper and Gradio

## 6. E2E ASR model finetune with Nemo
Quartznet Model finetune with Nemo(English to Korean)

## 7. WFST
Exercise for WFST using k2
- C,L,G transducer
- composition, determinization

## 8. E2E ASR model finetune with HuggingFace
Wav2Vec2.0 Model finetune with HunggingFace(English to Korean)

Whisper Model finetune with HunggingFace(English to Korean)

---------------
# Course Materials
## Chapter 1
1.	Huang, X. D., Acero, A., Hon, H. W., & Foreword By-Rabiner, L. (2001). Spoken Language Processing: A Guide to Theory, Algorithm, and System Development - Chapter 9. Pearson Education. (https://dl.acm.org/doi/book/10.5555/560905#cited-by-sec)
2.	Uday Kamath, John Liu, and James Whitaker (2019). Deep Learning for NLP and Speech Recognition. Springer. (https://www.amazon.com/Deep-Learning-NLP-Speech-Recognition/dp/3030145980)

## Chapter 4
1.	Introduction to Deep Learning: MIT 6.S191. (2023). Lecture2 – Deep Sequence Modeling. (http://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L2.pdf)

## Chapter 5: Sequence-to-Sequence with Attention
1.	Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. arXiv preprint arXiv:1409.3215. (https://arxiv.org/abs/1409.3215)
2.	Cho, K., Van Merriënboer, B., Bahdanau, D., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. arXiv preprint arXiv:1406.1078. (https://arxiv.org/abs/1406.1078)
3.	Stanford University. (2022). CS224N: Natural Language Processing with Deep Learning: Lecture7 – Machine Translation, Sequence-to-Sequence and Attention. (https://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture07-nmt.pdf)
4.	Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. arXiv preprint arXiv:1409.0473. (https://arxiv.org/abs/1409.0473)

## Chapter 6
1.	Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks. ICML'06: Proceedings of the 23rd international conference on Machine learning. (https://www.cs.toronto.edu/~graves/icml_2006.pdf)
2.	Olah, C., & Carter, S. (2017). Distill: Sequence Modeling With CTC. (https://distill.pub/2017/ctc/)
3.	Bluche, T. (N.D.). The intriguing blank label in CTC. [Blog post]. (https://www.tbluche.com/ctc_and_blank.html)

## Chapter 7: Transformer
1.	Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. arXiv preprint arXiv:1706.03762. (https://arxiv.org/abs/1706.03762)
2.	Olah, C. (2015). Understanding LSTM Networks. [Blog post]. (https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
3.	Stanford University. (2022). CS224N: Natural Language Processing with Deep Learning: Lecture9 – Transformers. (https://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture09-transformers.pdf)
4.	Jurafsky, D., & Martin, J. H. (2022). Speech and Language Processing: Chapter9 – RNNs and LSTMs. (https://web.stanford.edu/~jurafsky/slp3/)
5.	He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR. (https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
6.	Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. arXiv preprint arXiv:1607.06450. (https://arxiv.org/abs/1607.06450)
7.	Alammar, J. (N.D.). The Illustrated Transformer. [Blog post]. (https://jalammar.github.io/illustrated-transformer/)
8.	Bloem, P. (N.D.). TRANSFORMERS FROM SCRATCH. [Blog post]. (https://peterbloem.nl/blog/transformers)
9.	Gulati, A., Qin, J., Chiu, C. C., et al. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition. arXiv preprint arXiv:2005.08100. (https://arxiv.org/abs/2005.08100)

## Chapter 8
1.	Hinton, G., Deng, L., Yu, D., et al. (2012). Deep Neural Networks for Acoustic Modeling in Speech Recognition. IEEE Signal Processing Magazine. (https://www.cs.toronto.edu/~hinton/absps/DNN-2012-proof.pdf)
2.	Young, S. J., & Woodland, P. C. (1994). Tree-Based State Tying for High Accuracy Acoustic Modelling. Proc. ARPA Spoken Language Systems Technology Workshop. (https://aclanthology.org/H94-1062.pdf)
3.	Barsky, M. (N.D.). Victoria university: data mining Lab3 - Classifiers: toy example of decision tree. (http://csci.viu.ca/~barskym/teaching/DM_LABS/LAB_3/Lab3_decisiontreeexample.pdf)
4.	Bengio, Y., Lamblin, P., Popovici, D., & Larochelle, H. (2007). Greedy Layer-Wise Training of Deep Networks. Advances in Neural Information Processing Systems 19. (https://proceedings.neurips.cc/paper/2006/file/5da713a690c067105aeb2fae32403405-Paper.pdf)
5.	Graves, A., Mohamed, A. R., & Hinton, G. (2013). SPEECH RECOGNITION WITH DEEP RECURRENT NEURAL NETWORKS. 2013 IEEE International Conference on Acoustics, Speech and Signal Processing. (http://www.cs.toronto.edu/~hinton/absps/RNN13.pdf)

## Chapter 9
1.	American National Corpus (ANC). (N.D.). AMC. (https://anc.org/data/anc-second-release/frequency-data/)
2.	Sekine, S. (2010). On-Demand Information Extraction and Linguistic Knowledge Acquisition. New York University. (https://nlp.cs.nyu.edu/sekine/papers/10spring.pdf)
3.	Gillick, L., & Cox, S. J. (1991). Some statistical issues in the comparison of speech recognition algorithms. ICASSP-91: International Conference on Acoustics, Speech, and Signal Processing. (https://www.researchgate.net/publication/2360210_Comparison_Of_Part-Of-Speech_And_Automatically_Derived_Category-Based_Language_Models_For_Speech_Recognition)

## Chapter 10
1.	Mohri, M., Pereira, F., & Riley, M. (2008). SPEECH RECOGNITION WITH WEIGHTED FINITE-STATE TRANSDUCERS. Springer Handbook of Speech Processing. (https://cs.nyu.edu/~mohri/pub/hbka.pdf)
2.	Panayotov, V. (2012). Decoding graph construction in Kaldi: A visual walkthrough. [Blog post]. (http://vpanayotov.blogspot.com/2012/06/kaldi-decoding-graph-construction.html)
3.	Lecture on Weighted Finite State Transducers in Automatic Speech Recognition. (N.D.). Brno University of Technology. (http://www.fit.vutbr.cz/study/courses/ZRE/public/pred/13_wfst_sid_lid/zre_lecture_asr_wfst.pdf)

