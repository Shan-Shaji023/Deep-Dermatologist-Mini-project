<script>
    document.getElementById('myfile').addEventListener('change', function() {
        var reader = new FileReader();

        reader.onload = function(e) {
            document.getElementById('selected_image').setAttribute('src', e.target.result);
        }

        reader.readAsDataURL(this.files[0]);
    });
</script>
