input               = document.getElementById("text-input")
sentence_container  = document.getElementById("sentence-containter")
sentence_table_body = document.getElementById("sentence-table-body")

sentences = [
    "Ala ma kota",
    "Ula ma psa",
    "Lorem ipsum",
    "Przykładowe zdanie"
]

input.oninput = function(e) {
    if (input.value) {
        if(sentence_container.style.display === ""){
            populateTable(sentences);
        }
        sentence_container.style.display = "block";
      } else {
        sentence_container.style.display = "";
        clearNode(sentence_table_body)
      }
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
        })
        sentence_table_body.appendChild(tr)
    })
}