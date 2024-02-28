 function saveImage() {
   const fileInput = document.getElementById("myfile");
   const outputDiv = document.getElementById("output");
  
   if (fileInput.files.length === 0) {
     outputDiv.innerHTML = "<p>Please select a file</p>";
     return;
   }
  
   const file = fileInput.files[0];
 
   const reader = new FileReader();
   reader.readAsDataURL(file);
   reader.onload = function() {
     const img = new Image();
     img.src = reader.result;
     img.onload = function() {
       const canvas = document.createElement("canvas");
       canvas.width = img.width;
       canvas.height = img.height;
       const ctx = canvas.getContext("2d");
       ctx.drawImage(img, 0, 0);
       const link = document.createElement("a");
       link.download = file.name;
       link.href = canvas.toDataURL("image/png");
       link.click();
       outputDiv.innerHTML = "<p>Image saved successfully!</p>";
     }
   }
 }

