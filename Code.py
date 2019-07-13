# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:37:45 2019

@author: RamasamiXD
"""

import numpy as np
import tensorflow as tf
import re
import time


#Read the data
lines = open("cornell movie-dialogs corpus\movie_lines.txt", encoding = "utf-8", errors = "ignore").read().split("\n")
conversations = open("cornell movie-dialogs corpus\movie_conversations.txt", encoding = "utf-8", errors = "ignore").read().split("\n")

#Create Dictionary of lists
id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        
#Create a list of all conversations
conversation_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ","")
    conversation_ids.append(_conversation.split(","))
    
#Q&A
questions = []
answers = []
for conversation in conversation_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
      
#Clean the text
def cleantext(text):
    text = text.lower()
    text =  re.sub(r"i'm","i am", text)
    text =  re.sub(r"\'s"," is", text)
    text =  re.sub(r"\'re"," are", text)
    text =  re.sub(r"\'ve"," have", text)
    text =  re.sub(r"\'ll"," will", text)
    text =  re.sub(r"\'d"," would", text)
    text =  re.sub(r"won't","will not", text)
    text =  re.sub(r"&","and", text)
    text =  re.sub(r"[-()\"\'*#/@;:<>{}+=~!|\.?,]","", text)
    return text

clean_questions = []
clean_answers = []
for question in questions:
    clean_questions.append(cleantext(question))
for answer in answers:
    clean_answers.append(cleantext(answer)) 
    
#Word count
word2count = {}
for question in clean_questions:
    for word in question.split(" "):
        if word not in word2count:
            word2count[word]=0
        word2count[word]+=1
for answer in clean_answers:
    for word in answer.split(" "):
        if word not in word2count:
            word2count[word]=0
        word2count[word]+=1

#word2int
questionwords2int = {}
word_number = 0
threshold = 20

for word,count in word2count.items():
    if count > threshold:
        questionwords2int[word] = word_number
        word_number += 1

answerwords2int = {}
word_number = 0
for word,count in word2count.items():
    if count > threshold:
        answerwords2int[word] = word_number
        word_number += 1

#Adding additional tokens
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionwords2int[token] = len(questionwords2int)+1
for token in tokens:
    answerwords2int[token] = len(answerwords2int)+1

#Create reverse map
answerints2word = {w_i: w for w,w_i in answerwords2int.items()}
questionints2word = {w_i: w for w,w_i in questionwords2int.items()}

#Adding EOS to answers
for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"
    
#Convert to int
questions_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionwords2int:
            ints.append(questionwords2int['<OUT>'])
        else:
            ints.append(questionwords2int[word])
    questions_to_int.append(ints)
    
answers_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerwords2int:
            ints.append(answerwords2int['<OUT>'])
        else:
            ints.append(answerwords2int[word])
    answers_to_int.append(ints)
    
#Sort by length of question
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1,25+1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])
            

#Seq2Seq Model
            
#Creating placeholders
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None], name = 'input')
    targets = tf.placeholder(tf.int32, [None,None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')  #dropout rate
    return inputs, targets, lr, keep_prob

#Preprocessing the targets
def preprocessing_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets

#Encoder
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, Sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, cell_bw = encoder_cell, sequence_length = Sequence_length, inputs = rnn_inputs, dtype = tf.float32)
    return encoder_state

#Decoder Train
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.dynamic_rnn_decoder(decoder_cell, training_decoder_function, decoder_embedded_input, sequence_length, scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
    
#Decoder Test
def decode_test_set(encoder_state, decoder_cell, decoder_embedded_input, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.dynamic_rnn_decoder(decoder_cell, training_decoder_function, decoder_embedded_input, sequence_length, scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
    
    