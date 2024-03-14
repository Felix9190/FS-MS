import torch
# from sentence_transformers import SentenceTransformer
# from gensim.models import keyedvectors

# 编码单个单词的特征向量
# def encode_word(word, model):
#     # 使用BERT的分词器对单词进行编码
#     encoded_word = model.encode(word, convert_to_tensor=True)
#     return encoded_word

# 编码整个标签的特征向量并计算平均值
def encode_label(label, model, tokenizer):
    words = label.split("-")  # 使用"-"分割多个单词
    # encoded_words = [model.encode(word, model, tokenizer) for word in words]  # 对每个单词进行编码
    encoded_inputs_src = tokenizer(words, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs_src = model(**encoded_inputs_src)
    encoded_words = outputs_src.last_hidden_state[:, 0, :]  # (num_classess, 768)
    # encoded_label = torch.mean(torch.stack(encoded_words), dim=0)  # 计算编码后的单词向量的平均值
    encoded_label = torch.sum(encoded_words, dim=0)
    # encoded_label = torch.mean(encoded_words, dim=0)  # 计算编码后的单词向量的平均值

    return encoded_label

def encode_class_label(labels_list, model, tokenizer):
    # 对标签进行编码并计算平均值
    encoded_labels = [encode_label(label, model, tokenizer) for label in labels_list]
    semantic_vectors = torch.stack(encoded_labels)
    return semantic_vectors




