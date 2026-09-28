import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
const c=vm.createContext({Intl,Date,Number});
vm.runInContext(readFileSync(new URL('../assets/time.js',import.meta.url),'utf8'),c);
test('Eastern display follows summer EDT and winter EST rather than browser timezone',()=>{
 assert.match(c.easternTime('2026-09-28T15:23:00Z'),/11:23 AM EDT/);
 assert.match(c.easternTime('2026-01-28T15:23:00Z'),/10:23 AM EST/);
 assert.equal(c.easternDate('2026-09-28T01:00:00Z'),'Sep 27');
});
test('fall-back repeated hour has an unambiguous zone label',()=>{
 assert.match(c.easternTime('2026-11-01T05:30:00Z'),/01:30 AM EDT/);
 assert.match(c.easternTime('2026-11-01T06:30:00Z'),/01:30 AM EST/);
 assert.equal(c.easternTime('not a date'),'—');
});
