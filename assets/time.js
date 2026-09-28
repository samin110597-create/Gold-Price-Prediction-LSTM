// Display only. Stored timestamps remain UTC for causal joins and audits.
'use strict';
const easternZone='America/Toronto';
function easternTime(value){
 if(!value||!Number.isFinite(Date.parse(value)))return '—';
 return new Intl.DateTimeFormat('en-US',{timeZone:easternZone,year:'numeric',month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:true,timeZoneName:'short'}).format(new Date(value));
}
function easternDate(value){
 return new Intl.DateTimeFormat('en-US',{timeZone:easternZone,month:'short',day:'2-digit'}).format(new Date(value));
}
