// Simple form validation
document.addEventListener('DOMContentLoaded', function() {
    // Add basic form validation
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const inputs = this.querySelectorAll('input[required], textarea[required]');
            let valid = true;
            
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    valid = false;
                    input.style.borderColor = 'red';
                } else {
                    input.style.borderColor = '';
                }
            });
            
            if (!valid) {
                e.preventDefault();
                alert('Please fill in all required fields.');
            }
        });
    });
    
    // Add code editor features to textareas
    const codeTextareas = document.querySelectorAll('textarea[name="code_input"]');
    
    codeTextareas.forEach(textarea => {
        // Add a placeholder for code
        textarea.placeholder = textarea.placeholder || "Paste your Python code here...";
        
        // Add event listener for tab key
        textarea.addEventListener('keydown', function(e) {
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = this.selectionStart;
                const end = this.selectionEnd;
                
                // Insert tab character
                this.value = this.value.substring(0, start) + '    ' + this.value.substring(end);
                
                // Move cursor position
                this.selectionStart = this.selectionEnd = start + 4;
            }
        });
    });
});