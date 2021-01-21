input               = document.getElementById("text-input")
sentence_container  = document.getElementById("sentence-containter")
sentence_table_body = document.getElementById("sentence-table-body")

var sentences = []

input.oninput = onInputChange

function onInputChange(e){
    if (input.value) {
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
            sentences = xhr.response.sentences;

            clearNode(sentence_table_body);
            populateTable(sentences);
        }
    };
    xhr.send(JSON.stringify({
        "text" : text
    }));
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
        td.innerText = item
        tr.appendChild(td)
        tr.onclick = (function(e){
            input.value = e.target.innerText
            onInputChange()
        })
        sentence_table_body.appendChild(tr)
    })
}