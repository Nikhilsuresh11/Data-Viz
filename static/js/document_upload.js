// Add this JavaScript to handle the document upload and URL scraping process

document.addEventListener("DOMContentLoaded", () => {
    // Document upload form
    const documentUploadForm = document.getElementById("document-upload-form")
    const urlForm = document.getElementById("url-form")
    const tablesContainer = document.getElementById("tables-container")
    const loadingIndicator = document.getElementById("loading-indicator")
    const errorMessage = document.getElementById("error-message")
  
    if (documentUploadForm) {
      documentUploadForm.addEventListener("submit", (e) => {
        e.preventDefault()
  
        const fileInput = document.getElementById("document-file")
        if (!fileInput.files.length) {
          showError("Please select a file to upload.")
          return
        }
  
        const file = fileInput.files[0]
        const formData = new FormData()
        formData.append("file", file)
  
        // Show loading indicator
        showLoading("Extracting tables from document...")
  
        // Send the file to the server
        fetch("/extract-tables-from-document", {
          method: "POST",
          body: formData,
        })
          .then((response) => response.json())
          .then((data) => {
            hideLoading()
  
            if (data.error) {
              showError(data.error)
              return
            }
  
            if (data.success && data.tables && data.tables.length > 0) {
              displayTables(data.tables)
            } else {
              showError("No tables found in the document.")
            }
          })
          .catch((error) => {
            hideLoading()
            showError("Error extracting tables: " + error.message)
          })
      })
    }
  
    if (urlForm) {
      urlForm.addEventListener("submit", (e) => {
        e.preventDefault()
  
        const urlInput = document.getElementById("url-input")
        if (!urlInput.value.trim()) {
          showError("Please enter a URL.")
          return
        }
  
        const url = urlInput.value.trim()
  
        // Show loading indicator
        showLoading("Extracting tables from URL...<br>This may take a minute for complex websites.")
  
        // Send the URL to the server
        fetch("/extract-tables-from-url", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ url: url }),
        })
          .then((response) => response.json())
          .then((data) => {
            hideLoading()
  
            if (data.error) {
              showError(data.error)
              return
            }
  
            if (data.success && data.tables && data.tables.length > 0) {
              displayTables(data.tables)
            } else {
              showError("No tables found on the webpage.")
            }
          })
          .catch((error) => {
            hideLoading()
            showError("Error extracting tables: " + error.message)
          })
      })
    }
  
    function displayTables(tables) {
      // Clear previous tables
      tablesContainer.innerHTML = ""
  
      // Create a heading
      const heading = document.createElement("h3")
      heading.textContent = `Found ${tables.length} table${tables.length !== 1 ? "s" : ""}`
      tablesContainer.appendChild(heading)
  
      // Create a description
      const description = document.createElement("p")
      description.textContent = "Select a table to use for analysis:"
      tablesContainer.appendChild(description)
  
      // Create a container for the tables
      const tablesGrid = document.createElement("div")
      tablesGrid.className = "tables-grid"
      tablesContainer.appendChild(tablesGrid)
  
      // Add each table
      tables.forEach((table, index) => {
        const tableCard = document.createElement("div")
        tableCard.className = "table-card"
        tableCard.dataset.index = index
        tableCard.dataset.sessionId = table.sessionId
  
        // Add table title
        const tableTitle = document.createElement("h4")
        tableTitle.textContent = table.title || `Table ${index + 1}`
        tableCard.appendChild(tableTitle)
  
        // Add table preview
        const tablePreview = document.createElement("div")
        tablePreview.className = "table-preview"
  
        // Create an actual HTML table for preview
        const htmlTable = document.createElement("table")
        htmlTable.className = "preview-table"
  
        // Add headers
        if (table.headers && table.headers.length > 0) {
          const thead = document.createElement("thead")
          const headerRow = document.createElement("tr")
  
          table.headers.forEach((header) => {
            const th = document.createElement("th")
            th.textContent = header
            headerRow.appendChild(th)
          })
  
          thead.appendChild(headerRow)
          htmlTable.appendChild(thead)
        }
  
        // Add data rows (limit to 5 for preview)
        if (table.data && table.data.length > 0) {
          const tbody = document.createElement("tbody")
  
          table.data.slice(0, 5).forEach((row) => {
            const tr = document.createElement("tr")
  
            row.forEach((cell) => {
              const td = document.createElement("td")
              td.textContent = cell
              tr.appendChild(td)
            })
  
            tbody.appendChild(tr)
          })
  
          htmlTable.appendChild(tbody)
        }
  
        tablePreview.appendChild(htmlTable)
        tableCard.appendChild(tablePreview)
  
        // Add row count
        const rowCount = document.createElement("p")
        rowCount.className = "row-count"
        rowCount.textContent = `${table.data ? table.data.length : 0} rows × ${table.headers ? table.headers.length : 0} columns`
        tableCard.appendChild(rowCount)
  
        // Add select button
        const selectButton = document.createElement("button")
        selectButton.className = "btn btn-primary select-table-btn"
        selectButton.textContent = "Use This Table"
        selectButton.addEventListener("click", () => {
          selectTable(index, table.sessionId)
        })
        tableCard.appendChild(selectButton)
  
        tablesGrid.appendChild(tableCard)
      })
  
      // Show the tables container
      tablesContainer.style.display = "block"
    }
  
    function selectTable(tableIndex, sessionId) {
      // Show loading indicator
      showLoading("Processing selected table...")
  
      // Send the selection to the server
      fetch("/process-selected-table", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          tableIndex: tableIndex,
          sessionId: sessionId,
        }),
      })
        .then((response) => response.json())
        .then((data) => {
          hideLoading()
  
          if (data.error) {
            showError(data.error)
            return
          }
  
          if (data.success && data.redirect) {
            // Redirect to the overview page
            window.location.href = data.redirect
          } else {
            showError("Error processing table.")
          }
        })
        .catch((error) => {
          hideLoading()
          showError("Error processing table: " + error.message)
        })
    }
  
    function showLoading(message) {
      // Hide error message if shown
      errorMessage.style.display = "none"
  
      // Update and show loading indicator
      loadingIndicator.innerHTML = `
              <div class="spinner-border text-primary" role="status">
                  <span class="visually-hidden">Loading...</span>
              </div>
              <p>${message}</p>
          `
      loadingIndicator.style.display = "flex"
    }
  
    function hideLoading() {
      loadingIndicator.style.display = "none"
    }
  
    function showError(message) {
      errorMessage.textContent = message
      errorMessage.style.display = "block"
    }
  })
  
  