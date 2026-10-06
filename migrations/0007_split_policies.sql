-- Reading and writing get separate rules.
--
-- One policy "USING (can see)" also governed UPDATE and DELETE: whoever could SEE a row could
-- delete it. With public documents that would let an anonymous visitor delete them, and any member
-- delete a colleague's shared document, if any SQL ever tried. Now:
--   SELECT                     visibility rules (public / org / private)
--   INSERT, UPDATE, DELETE     only your own rows (org and owner match the verified login)
DROP POLICY tenant_isolation ON documents;
DROP POLICY tenant_isolation ON chunks;

CREATE POLICY read_visible ON documents FOR SELECT
    USING (org_id = app_org() AND (
        visibility = 'public'
        OR (app_user() IS NOT NULL AND (visibility = 'org' OR owner_id = app_user()))));
CREATE POLICY write_own ON documents FOR ALL
    USING (org_id = app_org() AND owner_id = app_user())
    WITH CHECK (org_id = app_org() AND owner_id = app_user());

CREATE POLICY read_visible ON chunks FOR SELECT
    USING (org_id = app_org() AND (
        visibility = 'public'
        OR (app_user() IS NOT NULL AND (visibility = 'org' OR owner_id = app_user()))));
CREATE POLICY write_own ON chunks FOR ALL
    USING (org_id = app_org() AND owner_id = app_user())
    WITH CHECK (org_id = app_org() AND owner_id = app_user());
