var input = document.querySelector('input');
var preview = document.querySelector('.preview');

input.style.opacity = 0; input.addEventListener('change', updateImageDisplay); function updateImageDisplay() {
	while (preview.firstChild) {
		preview.removeChild(preview.firstChild);
	}

	var curFiles = input.files;
	if (curFiles.length === 0) {
		var para = document.createElement('p');
		para.textContent = 'No files currently selected for upload';
		preview.appendChild(para);
	} else {
		var list = document.createElement('ol');
		preview.appendChild(list);
		for (var i = 0; i < curFiles.length; i++) {
			var listItem = document.createElement('li');
			var para = document.createElement('p');
			if (validFileType(curFiles[i])) {
				para.textContent = 'File name ' + curFiles[i].name + ', file size ' + returnFileSize(curFiles[i].size) + '.';
				var image = document.createElement('img');
				image.src = window.URL.createObjectURL(curFiles[i]);

				listItem.appendChild(image);
				listItem.appendChild(para);

			} else {
				para.textContent = 'File name ' + curFiles[i].name + ': Not a valid file type. Update your selection.';
				listItem.appendChild(para);
			}

			list.appendChild(listItem);
		}
	}
} var fileTypes = [
	'image/jpeg',
	'image/pjpeg',
	'image/png'
]

function validFileType(file) {
	for (var i = 0; i < fileTypes.length; i++) {
		if (file.type === fileTypes[i]) {
			return true;
		}
	}

	return false;
} function returnFileSize(number) {
	if (number < 1024) {
		return number + 'bytes';
	} else if (number > 1024 && number < 1048576) {
		return (number / 1024).toFixed(1) + 'KB';
	} else if (number > 1048576) {
		return (number / 1048576).toFixed(1) + 'MB';
	}
}

$(function () {

	var dropbox = $('#dropbox'),
		message = $('.message', dropbox);

	var image = $('.image-file');

	image.on("change", function () {
		alert('Hello');
		console.log('Hello');
		// $('#mytext').val($(this).val());
	});

	dropbox.filedrop({
		paramname: 'file',
		maxfiles: 1,
		maxfilesize: 5,
		url: '/upload/',
		uploadFinished: function (i, file, response) {
			alert(file);
			console.log(response);
			$.data(file).addClass('done');
		},

		error: function (err, file) {
			switch (err) {
				case 'BrowserNotSupported':
					showMessage('Your browser does not support HTML5 file uploads!');
					break;
				case 'TooManyFiles':
					alert('Too many files! Please select ' + this.maxfiles + ' at most!');
					break;
				case 'FileTooLarge':
					alert(file.name + ' is too large! The size is limited to ' + this.maxfilesize + 'MB.');
					break;
				default:
					break;
			}
		},

		beforeEach: function (file) {
			if (!file.type.match(/^image\//)) {
				alert('Only images are allowed!');
				return false;
			}
		},

		uploadStarted: function (i, file, len) {
			createImage(file);
		},

		progressUpdated: function (i, file, progress) {
			$.data(file).find('.progress').width(progress);
		}

	});

	var template = '<div class="preview">' +
		'<span class="imageHolder">' +
		'<img />' +
		'<span class="uploaded"></span>' +
		'</span>' +
		'<div class="progressHolder">' +
		'<div class="progress"></div>' +
		'</div>' +
		'</div>';


	function createImage(file) {

		var preview = $(template),
			image = $('img', preview);

		var reader = new FileReader();

		image.width = 100;
		image.height = 100;

		reader.onload = function (e) {
			image.attr('src', e.target.result);
		};

		reader.readAsDataURL(file);

		message.hide();
		preview.appendTo(dropbox);

		$.data(file, preview);
	}

	function showMessage(msg) {
		message.html(msg);
	}

});
