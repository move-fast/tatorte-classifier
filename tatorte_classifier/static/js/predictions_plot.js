
var layout = {
    title: 'Categories:',
    font: { size: 18 }
};


var number2str;
Http = new XMLHttpRequest();
var url = "/api/categories";
Http.open("GET", url);
Http.send()
Http.onreadystatechange = function () {
    if (Http.readyState == 4 && Http.status == 200) {
        number2str = JSON.parse(Http.responseText)
    }
}
function getPredictions() {
    xhr = new XMLHttpRequest();
    var url = "/api/models/" + document.getElementById("model-id").innerHTML + "/predict";
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            x = [];
            y = [];
            predictions = JSON.parse(xhr.responseText)["predictions"]
            for (i of predictions) {
                x.push(number2str[i["category"]]["name"])
                y.push(i["probability"])
            }
            data = [{
                type: "bar",
                x: x,
                y: y,
                marker: {
                    color: "#2BBBAD",
                    line: {
                        width: 0.5
                    }
                }
            }]
            Plotly.newPlot('predictions_graph', data, layout, { responsive: true });
        }
    }
    var data = JSON.stringify({ "data": document.getElementById("data-text").value, "parameters": { "max_categories": 5 } });
    xhr.send(data);
}
