input               = document.getElementById("text-input")
sentence_container  = document.getElementById("sentence-containter")
sentence_table_body = document.getElementById("sentence-table-body")
spinner             = document.getElementById("spinner")
num                 = document.getElementById("num")
models              = document.getElementById("models")

numoldvalue = num.value

var sentences = []

num.oninput = onNumChange

function onNumChange(e){
    if(num.value > 20 || num.value < 1)
        if(num.value !== "")
            num.value = numoldvalue
   numoldvalue = num.value
}

input.oninput = onInputChange

function onInputChange(e){
    if (input.value.length > 1 && input.value[input.value.length - 1] == " " && input.value[input.value.length - 2] != " ") {
        if(sentences.length == 0){
            getSentences(input.value)
            return
        }
        if(!sentences.every(item => {
            return item.startsWith(input.value)
        })){
            getSentences(input.value)
        }
    } else {
        clearNode(sentence_table_body)
        sentences = []
    }
}

function getSentences(text){
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/generate");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.responseType = "json";
    xhr.onload = function() {
        if(xhr.status == 200){
            spinner.setAttribute("hidden", "")
            sentences = xhr.response.sentences;
            var newsentences = []
            sentences.forEach(sen =>{
                newsentences.push(unescape(sen));
            })
            sentences = newsentences;
            clearNode(sentence_table_body);
            populateTable(sentences);
        }
    };
    xhr.send(JSON.stringify({
        "text" : text,
        "num" : num.value,
        "model" : models.value
    }));
    spinner.removeAttribute("hidden")
}

function clearNode(node){
    while (node.firstChild){ 
        node.removeChild(node.lastChild); 
    }
}

function populateTable(list){
    list.forEach(function(item){
        tr = document.createElement("tr")
        td = document.createElement("td")
        td.innerHTML = item
        tr.appendChild(td)
        tr.onclick = (function(e){
            input.value = e.target.innerText + " "
            onInputChange()
        })
        sentence_table_body.appendChild(tr)
    })
}