

// Drag and drop controller
document.querySelectorAll(".drop-zone__input").forEach((inputElement) => {
    const dropZoneElement = inputElement.closest(".drop-zone");
    dropZoneElement.addEventListener("click", (e) => {
        inputElement.click();
    });
    inputElement.addEventListener("change", (e) => {
        if (inputElement.files.length) {
            updateThumbnail(dropZoneElement, inputElement.files[0]);
        }
    });
    dropZoneElement.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZoneElement.classList.add("drop-zone--over");
    });
    ["dragleave", "dragend"].forEach((type) => {
        dropZoneElement.addEventListener(type, (e) => {
            dropZoneElement.classList.remove("drop-zone--over");
        });
    });
    dropZoneElement.addEventListener("drop", (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            inputElement.files = e.dataTransfer.files;
            updateThumbnail(dropZoneElement, e.dataTransfer.files[0]);
        }
        dropZoneElement.classList.remove("drop-zone--over");
    });
});
  
function updateThumbnail(dropZoneElement, file) {
    let thumbnailElement = dropZoneElement.querySelector(".drop-zone__thumb");
    // First time - remove the prompt
    if (dropZoneElement.querySelector(".drop-zone__prompt")) {
        dropZoneElement.querySelector(".drop-zone__prompt").remove();
    }
    // First time - there is no thumbnail element, so lets create its
    if (!thumbnailElement) {
        thumbnailElement = document.createElement("img");
        thumbnailElement.classList.add("drop-zone__thumb");
        thumbnailElement.setAttribute("id", "file_upload");
        dropZoneElement.appendChild(thumbnailElement);
    }
    thumbnailElement.dataset.label = file.name;
    // Show thumbnail for image files
    if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            thumbnailElement.style.backgroundImage = `url('${reader.result}')`;
            thumbnailElement.src=`${reader.result}`;
        };
        image_dir = reader;
    } else {
        thumbnailElement.style.backgroundImage = null;
    }
}

// TF prediction
async function predict() {
    var imagePreview = document.getElementById("file_upload");
    if (!image_dir.result || !image_dir.result.startsWith("data")) {
      window.alert("Please select an image before submit.");
      return;
    }
    try {
        let tensorImg = tf.browser.fromPixels(imagePreview).resizeNearestNeighbor([100, 195]).expandDims();
        prediction = await model.predict(tensorImg).data();

        predResult.classList.remove('warning');
        predResult.classList.remove('unsure');
        predResult.classList.remove('safe');

        // console.log(prediction)
        if (prediction[0] === 0) {
            predResult.innerHTML = "No Wildfire Detected";
            predResult.classList.add('safe');
        } else if (prediction[0] === 1) {
            predResult.innerHTML = "Warning Potential Wildfire";
            predResult.classList.add('warning');
        } else {
            predResult.innerHTML = "There was an issue";
            predResult.classList.add('unsure');
        }
    } 
    catch(error){
        predResult.innerHTML = error.message;
        predResult.classList.add('unsure');
    }

    show(predResult)
}

var imageDisplay = document.getElementById("image-display");
var predResult = document.getElementById("pred-result2");
var model = undefined;

async function initialize() {
    model = await tf.loadLayersModel('static/model.json');
}

function clearImage() {
    imageDisplay.src = "";
    predResult.innerHTML = "";
    hide(imageDisplay);
    hide(predResult);
    imageDisplay.classList.remove("loading");
}

function previewFile(file) {
    var fileName = encodeURI(file.name);
    var reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = () => {
        predResult.innerHTML = "";
        imageDisplay.classList.remove("loading");
    };
}

function hide(el) {
    el.classList.add("hidden");
}

function show(el) {
    el.classList.remove("hidden");
}

// UI Controller
var UI_controller = (function() {
    var DOM_strings = {
        date_label: '.footer__date--year',
        nav_container: 'navbar__container',
        nav_toggler: 'navbar-toggler',
        filter_toggler: 'navbar__filter',
        nav_icon: 'navbar__icon',
        nav_logo: 'navbar__logo',
        item_1: 'nav-item--1',
        item_2: 'nav-item--2',
        item_3: 'nav-item--3',
        item_4: 'nav-item--4',
        item_5: 'nav-item--5',
        link_1: 'nav-link--1',
        link_2: 'nav-link--2',
        link_3: 'nav-link--3',
        link_4: 'nav-link--4',
        link_5: 'nav-link--5',
    };
    return { 
        display_method: function() {
            var now, year;
            now = new Date();
            year = now.getFullYear();
            document.querySelector(DOM_strings.date_label).textContent = year;
        },
        dark_nav: function() {
            document.getElementById(DOM_strings.nav_container).className = "navbar__container navbar__container--white";
            document.getElementById(DOM_strings.nav_toggler).className = "navbar-toggler navbar-toggler--dark";
            document.getElementById(DOM_strings.nav_icon).className = "navbar__icon navbar__icon--dark";
            document.getElementById(DOM_strings.nav_logo).src = "img/logo-black.png";
            document.getElementById(DOM_strings.item_1).className = "nav-item nav-item--dark";
            document.getElementById(DOM_strings.item_2).className = "nav-item nav-item--dark";
            document.getElementById(DOM_strings.item_3).className = "nav-item nav-item--dark";
            document.getElementById(DOM_strings.item_4).className = "nav-item nav-item--dark";
            document.getElementById(DOM_strings.item_5).className = "nav-item nav-item--dark";
            document.getElementById(DOM_strings.link_1).className = "nav-link nav-link--dark";
            document.getElementById(DOM_strings.link_2).className = "nav-link nav-link--dark";
            document.getElementById(DOM_strings.link_3).className = "nav-link nav-link--dark";
            document.getElementById(DOM_strings.link_4).className = "nav-link nav-link--dark";
            document.getElementById(DOM_strings.link_5).className = "nav-link nav-link--dark";
        },
        light_nav: function() {
            document.getElementById(DOM_strings.nav_container).className = "navbar__container";
            document.getElementById(DOM_strings.nav_toggler).className = "navbar-toggler";
            document.getElementById(DOM_strings.nav_icon).className = "navbar__icon";
            document.getElementById(DOM_strings.nav_logo).src = "img/logo-white.png";
            document.getElementById(DOM_strings.item_1).className = "nav-item";
            document.getElementById(DOM_strings.item_2).className = "nav-item";
            document.getElementById(DOM_strings.item_3).className = "nav-item";
            document.getElementById(DOM_strings.item_4).className = "nav-item";
            document.getElementById(DOM_strings.item_5).className = "nav-item";
            document.getElementById(DOM_strings.link_1).className = "nav-link";
            document.getElementById(DOM_strings.link_2).className = "nav-link";
            document.getElementById(DOM_strings.link_3).className = "nav-link";
            document.getElementById(DOM_strings.link_4).className = "nav-link";
            document.getElementById(DOM_strings.link_5).className = "nav-link";
        },
        open_nav: function() {
            dark_nav()
        },
        get_DOM_strings: function() {
            return DOM_strings;
        },
    };
})();

// Event Controller
var event_controller = (function() {
    return {
        scroll_event: function() { 
            return document.documentElement.scrollTop
        },
        get_toggler: function(DOM) { 
            let elements = document.getElementsByClassName(DOM.nav_toggler);
            result = elements[0].getAttribute('aria-expanded');
            if (result === 'true') {
                return true;
            } else {
                return false;
            }
        },
    };
})();

// Global App Controller
var app_controller = (function(event_ctr, UI_ctr) { 
    let DOM = UI_ctr.get_DOM_strings();
    let change_val = parseInt(document.getElementsByClassName('header')[0].getAttribute('header-change'));
    let clicks = 1
    var setup_event_listeners = function() {
        window.onscroll = function() {
            navbar_handler();
        };
    };
    var navbar_handler = function() {
        let scroll_top = event_ctr.scroll_event();
        if (change_val == 0) {
            UI_ctr.dark_nav();
            document.getElementById(DOM.filter_toggler).className = "navbar__filter navbar__filter--closed";            
        }
        if ((clicks % 2) === 0) {
            UI_ctr.dark_nav();
            document.getElementById(DOM.filter_toggler).className = "navbar__filter navbar__filter--open";
            document.getElementById(DOM.nav_icon).className = "navbar__icon navbar__icon--dark navbar__icon--clicked";
        } else if ((clicks % 2) !== 0) {
            if (scroll_top > change_val) {
                UI_ctr.dark_nav();
                document.getElementById(DOM.filter_toggler).className = "navbar__filter navbar__filter--closed";
            } else if (scroll_top < change_val) {
                UI_ctr.light_nav();
                document.getElementById(DOM.filter_toggler).className = "navbar__filter navbar__filter--closed";
            }
        }
    };
    return {
        init: function() {
            UI_ctr.display_method();
            setup_event_listeners();
            navbar_handler();
            initialize();
        },
        ctrl_nav_toggle: function() {
            clicks += 1;
            navbar_handler();
        }
    };
})(event_controller, UI_controller);
app_controller.init();