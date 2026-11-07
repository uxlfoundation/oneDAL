# Build date (YYYYMMDD) of the version header and the release header. Stamped
# builds (`--stamp`, set by release-dpc) read the workspace status instead.

DEV_BUILD_DATE = "21990101"

_PLACEHOLDER = "@ONEDAL_BUILD_DATE@"

def _build_date_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name)
    if ctx.attr.stamp != 1:
        # Constant content: no input depends on the clock or the status file.
        ctx.actions.write(out, DEV_BUILD_DATE + "\n")
        return [DefaultInfo(files = depset([out]))]
    ctx.actions.run_shell(
        inputs = [ctx.info_file],
        outputs = [out],
        command = """
d=$(grep '^STABLE_ONEDAL_BUILD_DATE ' "$1" | cut -d' ' -f2 | tr -d '\r')
case "$d" in
  [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]) printf '%s\\n' "$d" > "$2" ;;
  *) echo "$1: no STABLE_ONEDAL_BUILD_DATE, build with --workspace_status_command" >&2; exit 1 ;;
esac
""",
        arguments = [ctx.info_file.path, out.path],
    )
    return [DefaultInfo(files = depset([out]))]

build_date = rule(
    implementation = _build_date_impl,
    doc = "Writes the build date file: STABLE_ONEDAL_BUILD_DATE when stamped, DEV_BUILD_DATE otherwise.",
    attrs = {
        "stamp": attr.int(
            values = [-1, 0, 1],
            default = -1,
            doc = "1 reads the workspace status; 0 and -1 (the BUILD select maps --stamp to 1) use DEV_BUILD_DATE.",
        ),
    },
)

def fill_build_date(ctx, template, date_file, out):
    """Replaces the `@ONEDAL_BUILD_DATE@` token of `template` with the date in `date_file`."""
    ctx.actions.run_shell(
        inputs = [template, date_file],
        outputs = [out],
        command = 'd=$(cat "$2") && sed "s/@ONEDAL_BUILD_DATE@/$d/g" "$1" > "$3"',
        arguments = [template.path, date_file.path, out.path],
    )
