document.getElementById("analyzeForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  document.getElementById("loader").style.display = "block";

  const body = {
    person_name: document.getElementById("person_name").value,
    base_url: document.getElementById("base_url").value,
    start_date: document.getElementById("start_date").value,
    end_date: document.getElementById("end_date").value
  };

  const res = await fetch("/api/final_result", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  const data = await res.json();
  document.getElementById("loader").style.display = "none";

  if (data.status === "success") {
    localStorage.setItem("final_result", JSON.stringify(data.final_result));
    window.location.href = "/result";
  } else {
    alert("No data found or API failed!");
  }
});
