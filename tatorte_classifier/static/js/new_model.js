
function trainModel() {
    classifier = document.getElementById("clf").value
    vect_params = { "ngram_range": [parseInt(document.getElementById("ngram_1").value), parseInt(document.getElementById("ngram_2").value)] }
    values_per_category = parseInt(document.getElementById("values_per_category").value)
    test_size = parseFloat(document.getElementById("test_size").value)
    formElement = document.getElementById("clf_options")
    clf_options_iterator = new URLSearchParams(new FormData(formElement)).entries()
    var clf_params = {}
    for (var i of clf_options_iterator) {
        if (parseFloat(i[1]) == i[1]) {
            clf_params[i[0]] = parseFloat(i[1])
        } else if (parseInt(i[1]) == i[1]) {
            clf_params[i[0]] = parseInt(i[1])
        } else if (i[0] == "hidden_layer_sizes") {
            clf_params[i[0]] = i[1].split(" ").map(Number)
        } else {
            clf_params[i[0]] = i[1]
        }
    }
    params = { "clf": classifier, "vect_params": vect_params, "clf_params": clf_params, "test_size": test_size, "values_per_category": values_per_category }
    console.log(params)

    const Http = new XMLHttpRequest();
    Http.open("POST", "/api/models/");
    Http.setRequestHeader("Content-type", "application/json");
    Http.send(JSON.stringify(params));
}

function changeOptions() {
    document.getElementById("clf_options").innerHTML = ""
    const Http = new XMLHttpRequest();
    model_name = document.getElementById("clf").value
    const url = '/api/model_options/' + model_name;
    Http.open("GET", url);
    Http.send()
    Http.onreadystatechange = function () {
        if (this.status == 200 && this.readyState == 4) {
            options = JSON.parse(Http.response)
            console.log(options)
            for (var key in options) {
                console.log(typeof options[key])
                if (typeof options[key] == "number") {
                    document.getElementById("clf_options").innerHTML += "<div class='input-field col s6'><label>" + key + "</label><input type='number' name='" + key + "'step='0.000001' value=" + options[key] + "></div>"
                } else if (typeof options[key] == "object") {

                    document.getElementById("clf_options").innerHTML += "<select class='browser-default' name='" + key + "' id='dropdown-" + key + "'></select>"
                    for (i = 0; i < options[key].length; i++) {
                        console.log(options[key][i])
                        document.getElementById("dropdown-" + key).innerHTML += "<option value='" + options[key][i] + "'>" + options[key][i] + "</option>"
                    }
                } else {
                    document.getElementById("clf_options").innerHTML += "<div class='input-field col s6'><label>" + key + "</label><input type='text' name='" + key + "'value='" + options[key] + "'></div>"
                }
            }
        }
    }
}