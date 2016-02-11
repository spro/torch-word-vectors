require 'nn'

sentences = {}
words = {}
words[''] = 1 -- end marker
n_words = 1 -- Because of end marker

inputs = {}
outputs = {}

-- Read sentences and scores from data file

for line in io.lines('data.txt') do
    rating = line:split(':')
    sentence = rating[1]:lower()
    score = tonumber(rating[2])

    _, num_words = sentence:gsub("%w+", "")
    if num_words > 0 then
        print(sentence, score)
        sentences[#sentences + 1] = sentence
        table.insert(outputs, score)
    end
end

-- Turn sentences into word index arrays for LookupTable

function sentence_to_indexes(sentence)
    input_indexes = {}
    for word in sentence:gmatch("%w+") do
        if not words[word] then
            n_words = n_words + 1
            words[word] = n_words
        end
        table.insert(input_indexes, words[word])
    end
    table.insert(input_indexes, 1)
    return input_indexes
end

for i = 1, #sentences do
    inputs[i] = torch.FloatTensor(sentence_to_indexes(sentences[i]))
end

-- Build the network

embed_size = 2

net = nn.Sequential()
lookup = nn.LookupTable(n_words, embed_size)
net:add(lookup)
net:add(nn.Sum()) -- Sum of word vectors
net:add(nn.Sigmoid())

-- Build the dataset

dataset = {}
function dataset:size() return #sentences end

for i, sentence in pairs(sentences) do
    input = inputs[i]
    target = torch.zeros(2)
    print(outputs[i])
    local score = outputs[i]
    target[score+1] = 1
    dataset[i] = {input, target}
end

-- Train the network

criterion = nn.BCECriterion() -- Works well with sigmoid
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.1
trainer.learningRateDecay = 0.0001
trainer.maxIteration = 1000

print("Training...")

trainer:train(dataset)

-- Print the network's output of a given sentence

function test(sentence)
    output = net:forward(torch.Tensor(sentence_to_indexes(sentence)))
    print(string.format("%s ==> <%.4f, %.4f>", sentence, output[1], output[2]))
end

test("is good")
test("is great")
test("so great")
test("is bad")
test("really bad")
test("just so")

