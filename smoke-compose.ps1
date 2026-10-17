$ErrorActionPreference='Stop'
$base='http://127.0.0.1:3000/api'
$jobId=$null
$keyLine=Get-Content (Join-Path $PSScriptRoot '.env') | Where-Object { $_ -like 'API_KEY=*' }
$keyValue=$keyLine.Substring('API_KEY='.Length)
try {
 $headers=@{'X-API-Key'=$keyValue}
 $http=[Net.Http.HttpClient]::new();$http.DefaultRequestHeaders.Add('X-API-Key',$keyValue)
 $form=[Net.Http.MultipartFormDataContent]::new()
 $bytes=New-Object byte[] 512
 [byte[]]$sig=0,0,0,24,102,116,121,112,105,115,111,109
 [Array]::Copy($sig,$bytes,$sig.Length)
 $part=[Net.Http.ByteArrayContent]::new($bytes);$part.Headers.ContentType=[Net.Http.Headers.MediaTypeHeaderValue]::new('video/mp4');$form.Add($part,'video','smoke-test.mp4')
 $response=$http.PostAsync("$base/jobs",$form).GetAwaiter().GetResult()
 $text=$response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
 if([int]$response.StatusCode -ne 202){throw "Upload failed: $($response.StatusCode)"}
 $jobId=($text|ConvertFrom-Json).job_id
 if($jobId -notmatch '^[a-f0-9]{32}$'){throw 'Invalid job ID'}
 $deadline=(Get-Date).AddSeconds(20)
 do { Start-Sleep -Milliseconds 500; $job=Invoke-RestMethod "$base/jobs/$jobId" -Headers $headers } while($job.status -in @('queued','processing') -and (Get-Date) -lt $deadline)
 if($job.status -ne 'done' -or !$job.demo -or $job.violations.Count -eq 0){throw 'Demo analysis failed'}
 $foreign=Invoke-WebRequest "$base/jobs/$jobId" -Headers @{'X-API-Key'='invalid-key'} -SkipHttpErrorCheck
 if($foreign.StatusCode -ne 401){throw 'Invalid key accepted'}
 $removed=Invoke-WebRequest "$base/auth/register" -Method Post -SkipHttpErrorCheck
 if($removed.StatusCode -ne 404){throw 'Registration still enabled'}
 Write-Output 'PASS: API key, upload, PostgreSQL result, invalid key rejected, registration disabled'
} finally {
 if($jobId -match '^[a-f0-9]{32}$') { docker compose exec -T db psql -U cashguard -d cashguard -q -c "DELETE FROM jobs WHERE id='$jobId';"; docker compose exec -T backend rm -f "/data/uploads/$jobId.mp4" }
 if($form){$form.Dispose()};if($http){$http.Dispose()}
}
