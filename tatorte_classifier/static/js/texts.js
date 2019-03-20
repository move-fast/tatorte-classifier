function getAllCategories() {
    checkbox_1 = document.getElementById("checkbox-1").checked
    checkbox_2 = document.getElementById("checkbox-2").checked
    checkbox_3 = document.getElementById("checkbox-3").checked
    checkbox_4 = document.getElementById("checkbox-4").checked
    checkbox_5 = document.getElementById("checkbox-5").checked
    arr = [checkbox_1, checkbox_2, checkbox_3, checkbox_4, checkbox_5]
    if (checkbox_1 || checkbox_2 || checkbox_3 || checkbox_4 || checkbox_5) {
        var indexes = [], i = -1;
        while ((i = arr.indexOf(true, i + 1)) != -1) {
            indexes.push(i);
        }
        return indexes;
    } else {
        alert("At least one checkboxes has to be checked")
        return false
    }
}

function reload() {
    console.log(document.location.href)
    if (document.location.href.includes("change-data")) {
        document.location.href = "/texts/1"
    } else {
        location.reload()
    }
}

function getTextId() {
    if (document.location.href.includes("change-data")) {
        return document.getElementById("text-id").value
    } else {
        return document.getElementById("text-id").innerHTML
    }
}

function deleteAndReload(later_url) {
    const Http = new XMLHttpRequest();
    const url = '/api/texts/' + getTextId();
    Http.open("DELETE", url);
    Http.send();
    Http.onreadystatechange = (e) => {
        console.log(Http.responseText)
    }
    reload()
}
function submitAndReload() {
    const url = '/api/texts/' + getTextId();
    categories = getAllCategories()
    if (categories) {
        const Http = new XMLHttpRequest();
        Http.open("PATCH", url)
        Http.setRequestHeader("Content-type", "application/json")
        Http.send('{"categories": [' + categories + ']}')
        reload()
    }
}

function loadPreview() {
    if (getTextId() === 24) {
        document.location.href = "/change-data/" + getTextId()

    } else {
        alert("The given Id is not valid")
    }
}

function createAndReload() {
    const url = '/api/texts';
    data = document.getElementById("data-text").value
    categories = getAllCategories()
    if (categories) {
        const Http = new XMLHttpRequest();
        Http.open("POST", url)
        Http.setRequestHeader("Content-type", "application/json")
        Http.send('{"data": "' + data + '", "categories": [' + categories + ']}')
        location.reload()
    } else {
        alert("At least on checkbox has to be checked")
    }
}